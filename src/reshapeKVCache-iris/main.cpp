#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "utils.h"

#include "reference.h"
#include <type_traits>
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}


template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache(int num_tokens, int num_heads, int head_size,
                       int block_size, int x, int repeat)
{
    assert(head_size % x == 0);

    int key_stride = num_heads * head_size;
    int value_stride = key_stride;
    int headsize_div_x = head_size / x;

    uint64_t key_size = (uint64_t)num_tokens * key_stride;
    uint64_t value_size = key_size;

    uint64_t num_blocks = (num_tokens + block_size - 1) / block_size;

    uint64_t key_cache_size = num_blocks * block_size * key_stride;

    uint64_t value_cache_size = key_cache_size;

    std::vector<scalar_t> h_key(key_size);
    std::vector<scalar_t> h_value(value_size);
    std::vector<int64_t> h_slot_mapping(num_tokens);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    scalar_t key_max = 0, val_max = 0;
    for (uint64_t i = 0; i < key_size; i++) {
        h_key[i] = dist(gen);
        h_value[i] = dist(gen);
        key_max = std::max(key_max, h_key[i]);
        val_max = std::max(val_max, h_value[i]);
    }
    float k_scale = (float)key_max / 64.f;
    float v_scale = (float)val_max / 64.f;

    for (int i = 0; i < num_tokens; i++) {
        h_slot_mapping[i] = i;
    }

    std::shuffle(h_slot_mapping.begin(), h_slot_mapping.end(), gen);

    std::vector<cache_t> h_key_cache(key_cache_size, 0), h_value_cache(value_cache_size, 0);
    iris::DMem d_key(h_key.data(),key_size*sizeof(scalar_t)),d_value(h_value.data(),value_size*sizeof(scalar_t)),
      d_key_cache(h_key_cache.data(),key_cache_size*sizeof(cache_t)),d_value_cache(h_value_cache.data(),value_cache_size*sizeof(cache_t)),
      d_slot_mapping(h_slot_mapping.data(),num_tokens*sizeof(int64_t));
    const int device=hip_device();
    size_t lws=std::min(num_heads*headsize_div_x,512),gws=size_t(num_tokens)*lws;
    auto launch=[&](){
      void* args[]={&d_key,&d_value,&d_key_cache,&d_value_cache,&d_slot_mapping,&key_stride,&value_stride,&num_heads,&head_size,&block_size,&x,&k_scale,&v_scale};
      int info[]={iris_r,iris_r,iris_rw,iris_rw,iris_r,sizeof(key_stride),sizeof(value_stride),sizeof(num_heads),sizeof(head_size),sizeof(block_size),sizeof(x),sizeof(k_scale),sizeof(v_scale)};
      iris::Task task;check(task.kernel(std::is_same_v<scalar_t,float>?"reshape_float":"reshape_bf16",1,nullptr,&gws,&lws,13,args,info));check(task.submit(device,nullptr,true));
    };
    auto flush=[&](){iris::Task task;check(task.flush_out(d_key_cache));check(task.flush_out(d_value_cache));check(task.submit(device,nullptr,true));};
    // warmup
    for (int i = 0; i < 30; i++) {
        launch();
    }

    std::vector<cache_t> r_key_cache(key_cache_size, 0);
    std::vector<cache_t> r_value_cache(value_cache_size, 0);

    reference<scalar_t, cache_t, kv_dt>(
      h_key.data(), h_value.data(),
      r_key_cache.data(), r_value_cache.data(),
      h_slot_mapping.data(),
      key_stride, value_stride, num_tokens, num_heads, head_size, block_size,
      x, k_scale, v_scale);


    flush();

    auto verify=[&](){
    bool ok = true;
    for (uint64_t i = 0; i < key_cache_size; i++) {
      if (r_key_cache[i] != h_key_cache[i]) {
        printf("key %zu: %d %d\n", i, r_key_cache[i], h_key_cache[i]);
        ok = false;
        break;
      }
    }

    for (uint64_t i = 0; i < value_cache_size; i++) {
      if (r_value_cache[i] != h_value_cache[i]) {
        printf("value %zu: %d %d\n", i, r_value_cache[i], h_value_cache[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    uint64_t hash=1469598103934665603ULL;
    for(auto v:h_key_cache){hash^=v;hash*=1099511628211ULL;}
    for(auto v:h_value_cache){hash^=v;hash*=1099511628211ULL;}
    printf("CACHE_HASH %016llx bytes %llu\n",(unsigned long long)hash,(unsigned long long)(key_cache_size+value_cache_size));
    if(!ok)throw std::runtime_error("cache verification failed");
    };
    verify();


    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        launch();
    }


    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of reshape and cache kernel: %f (us)\n", (time * 1e-3f) / repeat);

    flush();verify();

    uint64_t bytes = sizeof(cache_t) * (key_cache_size + value_cache_size) +
                     sizeof(int64_t) * num_tokens +
                     sizeof(scalar_t) * (key_size + value_size);
    printf("Average bandwidth of reshape and cache kernel: %f (GB/s)\n", bytes * repeat * 1.0 / time);

}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <number of tokens> <number of heads> <head size> <repeat>\n", argv[0]);
        return 1;
    }
    const int num_tokens = atoi(argv[1]);
    const int num_heads = atoi(argv[2]);
    const int head_size = atoi(argv[3]);
    const int repeat = atoi(argv[4]);

    iris::Platform platform;check(platform.init(&argc,&argv,true));
    if(num_tokens<=0||num_heads<=0||head_size<=0||head_size%8||repeat<=0)throw std::runtime_error("positive dimensions/repeat and head size multiple of 8 required");
    for (int block_size = 16; block_size <= 32; block_size *= 2) {
        for (int x = 1; x <= 8; x *= 2) {
            printf("Block size: %d, Head size divisor: %d\n", block_size, x);
            printf("─────────────────────────────────────────────────────\n");
            printf("Reshape FP32 -> FP8 KV cache\n");
            reshape_and_cache<float, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(
              num_tokens, num_heads, head_size, block_size, x, repeat);

            printf("Reshape BF16 -> FP8 KV cache\n");
            reshape_and_cache<__hip_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(
              num_tokens, num_heads, head_size, block_size, x, repeat);
            printf("─────────────────────────────────────────────────────\n");
        }
    }

    check(platform.finalize());return 0;
}
