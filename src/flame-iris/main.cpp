/*
 * (c) 2009-2010 Christoph Schied <Christoph.Schied@uni-ulm.de>
 *
 * This file is part of flame.
 *
 * flame is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * flame is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with flame.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iris/iris.hpp>
#include <vector>
#include <stdexcept>
struct alignas(8) float2 {float x,y;};
struct float3 {float x,y,z;};
struct alignas(4) short2 {short x,y;};
static_assert(sizeof(float2)==8 && sizeof(float3)==12 && sizeof(short2)==4);
static int gpu_device=-1;
struct Param {void* ptr;int info;};
static Param param(iris::DMem& mem){return {&mem,iris_rw};}
static Param param(const iris::DMem& mem){return {const_cast<iris::DMem*>(&mem),iris_r};}
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

#include "flame.hpp"

#define VARPARM(a,b) const_mem_params.variation_parameters[a][b]
#define FUNC_COLOR(a) const_mem_params.function_colors[a]
#define FUNC_WEIGHT(a) function_weights[a]
#define AFFINE_PRE(a, b) const_mem_params.pre_transform_params[a][b]
#define AFFINE_POST(a, b) const_mem_params.post_transform_params[a][b]



int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("reset parameters..\n");
  ConstMemParams const_mem_params;
  float function_weights[NUM_FUNCTIONS];

  memset(&const_mem_params, 0, sizeof const_mem_params);

  for(int i = 0; i < NUM_FUNCTIONS; i++) {
    function_weights[i] = 0.0f;
    const_mem_params.variation_parameters[i][0].factor = 1.0f;
    const_mem_params.pre_transform_params[i][0] = 1.0f;
    const_mem_params.pre_transform_params[i][4] = 1.0f;
    const_mem_params.post_transform_params[i][0] = 1.0f;
    const_mem_params.post_transform_params[i][4] = 1.0f;
  }

  function_weights[0] = 1.0f;

  AFFINE_PRE(0, 0) = 1.4f;
  AFFINE_PRE(0, 1) = 0.6f;

  AFFINE_PRE(1, 2) = 0.4f;

  AFFINE_PRE(2, 0) = 0.3f;
  AFFINE_PRE(2, 2) = 1.0f;

  FUNC_COLOR(0) = 1.0f;
  FUNC_COLOR(2) = 0.4f;

  FUNC_WEIGHT(1) = 0.5f;
  FUNC_WEIGHT(2) = 0.6f;

  const_mem_params.enable_sierpinski = 1;

  VARPARM(2, 0).idx = 9;
  VARPARM(0, 0).idx = 13;
  VARPARM(1, 0).idx = 5;
  VARPARM(1, 1).idx = 7;

  VARPARM(2, 0).factor = 1.0f;
  VARPARM(0, 0).factor = 1.0f;
  VARPARM(1, 0).factor = 1.0f;
  VARPARM(1, 1).factor = -0.3f;

  AFFINE_PRE(2, 2) = -1.0f;
  AFFINE_PRE(2, 5) = -1.0f;

  srand(2);
  unsigned mersenne_state[624];
  for(int i = 0; i < 624; i++) mersenne_state[i] = rand();
  for(int i = 0; i < 10000; i++) mersenne_twister(mersenne_state);

  printf("generating random numbers\n");
  float *rn_tmp = new float[NUM_RANDOMS];
  for(int i = 0; i < NUM_RANDOMS; i++)
    rn_tmp[i] = mersenne_twister(mersenne_state) * (1.0f / 4294967296.0f);



  printf("generating permutations\n");
  PermSortElement *to_sort = new PermSortElement[NUM_THREADS];

  unsigned short *perm_data = new unsigned short[NUM_THREADS * NUM_PERMUTATIONS];

  for(int i = 0; i < NUM_PERMUTATIONS; i++) {
    //if(!(i & 127)) printf(".");
    for(int j = 0; j < NUM_THREADS; j++) {
      to_sort[j].value = mersenne_twister(mersenne_state);
      to_sort[j].idx = j;
    }
    std::sort(to_sort, to_sort + NUM_THREADS);
    for(int j = 0; j < NUM_THREADS; j++) {
      perm_data[i * NUM_THREADS + j] = to_sort[j].idx;
    }
  }





  float2 *points_tmp = new float2[NUM_THREADS];
  for(int i = 0; i < NUM_THREADS; i++) {
    points_tmp[i].x = (float(i) / NUM_THREADS - 0.5f) * 2.0f;
    points_tmp[i].y = (radical_inverse(i, 2) - 0.5f) * 2.0f;
  }




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
  iris::DMem random_numbers(rn_tmp,NUM_RANDOMS*sizeof(float));
  iris::DMem permutations(perm_data,NUM_THREADS*NUM_PERMUTATIONS*sizeof(unsigned short));
  iris::DMem start_points(points_tmp,NUM_THREADS*sizeof(float2));
  std::vector<short2> short_host(NUM_THREADS);std::vector<short> color_host(NUM_THREADS);
  std::vector<float3> vertices_host(NUM_POINTS_PER_THREAD*NUM_THREADS);
  iris::DMem short_points(short_host.data(),short_host.size()*sizeof(short2));
  iris::DMem colors(color_host.data(),color_host.size()*sizeof(short));
  iris::DMem vertices(vertices_host.data(),vertices_host.size()*sizeof(float3));
  iris::DMem params_memory(&const_mem_params,sizeof(ConstMemParams));
  const iris::DMem& read_params=params_memory;
  printf("entering mainloop\n");
  auto start = std::chrono::steady_clock::now();

  int perm_pos = 0;
  for (int n = 0; n < repeat; n++) {
    float sum = 0.0f;
    for(int i = 0; i < NUM_FUNCTIONS; i++)
      sum += function_weights[i];

    int num_threads_sum = 0;
    for(int i = 0; i < NUM_FUNCTIONS; i++) {
      int num_threads = (function_weights[i] / sum) * (NUM_THREADS);
      const_mem_params.thread_function_mapping[i] = num_threads;
      num_threads_sum += num_threads;
    }
    const_mem_params.thread_function_mapping[0] += NUM_THREADS - num_threads_sum;

    for(int i = 1; i < NUM_FUNCTIONS; i++) {
      const_mem_params.thread_function_mapping[i] +=
        const_mem_params.thread_function_mapping[i - 1];
    }

    const size_t grid=NUM_THREADS/THREADS_PER_BLOCK, block=THREADS_PER_BLOCK;
    params_memory.update(&const_mem_params);

    launch("kernel_initialize",grid,block,short_points, colors, permutations,
        perm_pos++, start_points, random_numbers, read_params);

    for(int i = 0; i < NUM_ITERATIONS; i++) {
      launch("kernel_iterate",grid,block,short_points, colors, permutations,
          perm_pos++, random_numbers, read_params);
      perm_pos %= NUM_PERMUTATIONS;
    }

    launch("kernel_generate_points",grid,block,vertices,
        short_points, colors, permutations, perm_pos++, 
        random_numbers, read_params);

    perm_pos += NUM_POINTS_PER_THREAD - 1;
    perm_pos %= NUM_PERMUTATIONS;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total frame time is %.3f s\n", time * 1e-9);

  #ifdef DUMP
  // dump vertices
  float3 *pixels = new float3[NUM_POINTS_PER_THREAD * NUM_THREADS];
  fetch(vertices,pixels);

  for (int i = 0; i < NUM_POINTS_PER_THREAD * NUM_THREADS; i++)
    printf("%d x=%.1f y=%.1f color=%.1f\n", i, pixels[i].x, pixels[i].y, pixels[i].z);

  delete[] pixels;
  #endif


  if(const char* path=getenv("HECBENCH_FLAME_OUTPUT")) {
    fetch(vertices,vertices_host.data());
    FILE* fp=fopen(path,"wb");
    if(!fp || fwrite(vertices_host.data(),sizeof(float3),vertices_host.size(),fp)!=vertices_host.size())return 2;
    fclose(fp);
  }
  platform.finalize();
  delete[] rn_tmp;delete[] perm_data;delete[] points_tmp;delete[] to_sort;
  return 0;
}
