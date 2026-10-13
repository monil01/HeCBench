#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <stdexcept>
#define CEILDIV(x,y) (((x)+(y)-1)/(y))
static int gpu_device=-1;
struct Param {void* ptr;int info;};
static Param param(iris::DMem& mem){return {&mem,iris_rw};}
template<class T> static Param param(const T& value){return {const_cast<T*>(&value),sizeof(T)};}
template<class... Args> static void launch(const char* name,size_t grid,size_t block,Args&&... args) {
  if(block<1 || block>1024) throw std::runtime_error("block size must be 1..1024");
  Param list[]={param(args)...};void* p[sizeof...(args)];int info[sizeof...(args)];
  for(size_t i=0;i<sizeof...(args);++i){p[i]=list[i].ptr;info[i]=list[i].info;}
  size_t g[]={grid*block},l[]={block};iris::Task task;
  if(task.kernel(name,1,nullptr,g,l,sizeof...(args),p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("kernel failed");
}
static void fetch(iris::DMem& mem,void* host) {
  iris::Task task;task.flush_out(mem);
  if(task.submit(gpu_device,nullptr,1) || mem.fetch(host)) throw std::runtime_error("fetch failed");
}

static void launch2(const char* name,size_t grid_y,size_t block,void** params,int* info,int nparams) {
  size_t g[]={block,grid_y},l[]={block,1};
  iris::Task task;
  if(task.kernel(name,2,nullptr,g,l,nparams,params,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("sort task failed");
}
void moe_align_block_size(iris::DMem& topk_ids,int num_experts,int block_size,
  iris::DMem& sorted_token_ids,iris::DMem& experts_ids,iris::DMem& total,void* expert_map,
  int numel,int topk,int maxpad) {
  int* map=nullptr;
  if(expert_map) throw std::runtime_error("CLI never supplies expert_map");
  const int warp=64; // Original HIP MI100 wave width.
  const int padded=CEILDIV(num_experts,warp)*warp;
  if(padded>=1024) throw std::runtime_error("expert count exceeds original limit");
  if(numel<1024 && num_experts<=64) {
    launch("align_small",1,256+std::max(num_experts,warp),topk_ids,sorted_token_ids,experts_ids,total,
      map,num_experts,block_size,numel,maxpad,topk);
  } else {
    std::vector<int> values(num_experts+1);
    iris::DMem cumsum(values.data(),values.size()*sizeof(int));
    launch("align_large",2,1024,topk_ids,sorted_token_ids,experts_ids,total,map,
      num_experts,padded,warp,block_size,numel,cumsum,maxpad,topk);
    void* params[]={&topk_ids,&sorted_token_ids,&cumsum,&map,&numel,&num_experts,&maxpad,&topk};
    int info[]={iris_r,iris_rw,iris_rw,sizeof(map),sizeof(numel),sizeof(num_experts),sizeof(maxpad),sizeof(topk)};
    launch2("sort_tokens",std::min(CEILDIV(numel,256),65535),256,params,info,8);
  }
}
void randperm(int* topk_ids, int topk, int m, int n) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i) v[i] = i;
  std::mt19937 gen(19937);
  for (int i = 0; i < m; i++) { 
    std::shuffle(v.begin(), v.end(), gen);
    for (int j = 0; j < topk; j++) {
      topk_ids[i * topk + j] = v[j];
      #ifdef DEBUG
      printf("%d ", v[j]);    
      #endif
    }
    #ifdef DEBUG
    printf("\n");
    #endif
  }
}


int main(int argc, char* argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  gpu_device=device;
  FILE* artifact=nullptr;
  if(const char* path=getenv("HECBENCH_MOE_OUTPUT")){artifact=fopen(path,"wb");if(!artifact)return 2;}

#ifdef DEBUG
   int tokens[] = {4};
   int experts[] = {4};
   int topks[] = {3};
   int block_sizes[] = {4};
#else
   int tokens[] = {1, 3, 256, 4096, 8192};
   int experts[] = {32, 128};
   int topks[] = {2, 3, 4};
   int block_sizes[] = {32};
#endif
   bool pad_sorted_ids = true;

   int *h_topk_ids;

   for (int i = 0; i < int(sizeof(tokens) / sizeof(tokens[0])); i++) {
     int m = tokens[i]; 

     for (int e = 0; e < int(sizeof(experts) / sizeof(experts[0])); e++) {
       int num_experts = experts[e]; 

       for (int k = 0; k < int(sizeof(topks) / sizeof(topks[0])); k++) {
         int topk = topks[k]; 
         int block_size = block_sizes[0];

         int topk_ids_size = m * topk;
         h_topk_ids = (int*) malloc(topk_ids_size * sizeof(int));
         randperm(h_topk_ids, topk, m, num_experts);

         iris::DMem topk_ids(h_topk_ids,topk_ids_size*sizeof(int));
         
         int max_num_tokens_padded = topk_ids_size + num_experts * (block_size - 1);

         if (pad_sorted_ids)
             max_num_tokens_padded = CEILDIV(max_num_tokens_padded, block_size) * block_size;

         if (topk_ids_size < num_experts)
             max_num_tokens_padded = std::min(topk_ids_size * block_size, max_num_tokens_padded);

         #ifdef DEBUG
         printf("max_num_tokens_padded: %zu\n",  max_num_tokens_padded);
         #endif

         std::vector<int> sorted_host(max_num_tokens_padded);
         iris::DMem sorted_ids(sorted_host.data(),max_num_tokens_padded*sizeof(int));


         int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
         std::vector<int> expert_host(max_num_m_blocks);
         iris::DMem expert_ids(expert_host.data(),max_num_m_blocks*sizeof(int));

         int total_host=0;
         iris::DMem num_tokens_post_pad(&total_host,sizeof(int));

         //bool ignore_invalid_experts = false;

         auto start = std::chrono::steady_clock::now();

         for (int n = 0; n < repeat; n++) {
           moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                nullptr,
                topk_ids_size,
                topk,
                max_num_tokens_padded 
           );
         }

         auto end = std::chrono::steady_clock::now();
         auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
         printf("Average execution time of the kernels (tokens %d, topk: %d, expert: %d, block_size %d): %f (us)\n",
                m, topk, num_experts, block_size, (time * 1e-3f) / repeat);

         //if expert_map is not None and not ignore_invalid_experts:
             //expert_ids = expert_map[expert_ids]

         //sorted_ids, expert_ids, num_tokens_post_pad
         int actual_num_tokens;
         fetch(num_tokens_post_pad,&actual_num_tokens);

         int *actual_expert_ids = (int*) malloc(max_num_m_blocks * sizeof(int));
         fetch(expert_ids,actual_expert_ids);

         int *actual_sorted_ids;
         actual_sorted_ids = (int*) malloc (max_num_tokens_padded * sizeof(int));
         fetch(sorted_ids,actual_sorted_ids);

         #ifdef DEBUG
         printf("actual sorted ids: ");
         for (int i = 0; i < max_num_tokens_padded; i++) {
           printf("%d ", actual_sorted_ids[i]);
         }

         printf("\nactual expert ids: ");
         for (int i = 0; i < max_num_m_blocks; i++) {
           printf("%d ", actual_expert_ids[i]);
         }
         printf("\nactual number of tokens: %d\n", actual_num_tokens);
         #endif

         bool ok = true;
         if (actual_num_tokens % block_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be divisible by block_size\n");
         }
         if (actual_num_tokens < topk_ids_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be at least total_tokens\n");
         }
         for (int i = 0; i < max_num_m_blocks; i++) {
           if (actual_expert_ids[i] < 0 || actual_expert_ids[i] >= num_experts) {
             ok = false;
             printf("Error: expert_ids should contain valid expert indices\n");
             break;
           }
         }

         int ei = 0;
         for (int t = 0; t < max_num_tokens_padded; t += block_size) {
           int eid = actual_expert_ids[ei++];
           for (int b = 0; b < block_size; b++) { 
             int v = actual_sorted_ids[t+b];
             if (v == topk_ids_size) {
               continue;
             }
             if (eid != h_topk_ids[v]) {
               ok = false;
               break;
             }
           }
         }
         printf("%s\n", ok ? "PASS" : "FAIL");
         if(!ok)return 1;
         if(artifact) {
           int header[]={m,num_experts,topk,block_size,max_num_tokens_padded,max_num_m_blocks,actual_num_tokens};
           fwrite(header,sizeof(int),7,artifact);
           fwrite(h_topk_ids,sizeof(int),topk_ids_size,artifact);
           fwrite(actual_sorted_ids,sizeof(int),max_num_tokens_padded,artifact);
           fwrite(actual_expert_ids,sizeof(int),max_num_m_blocks,artifact);
         }



         free(h_topk_ids);
         free(actual_sorted_ids);
         free(actual_expert_ids);
       }
     }
   }
   if(artifact)fclose(artifact);
   platform.finalize();
   return 0;
}
