//  Created by gangliao on 12/22/14.
//  Copyright (c) 2014 gangliao. All rights reserved.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "iris_support.hpp"
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <climits>
#include <chrono>







#define MAX_ALPHA 26

// number of threads in a block
#define BLOCK_SIZE 256


int device;
struct Buffer {
  std::vector<int> host;
  iris::DMem memory;
  explicit Buffer(size_t count):host(count,0),memory(host.data(),count*sizeof(int)) {}
};
void run_kernel(const char* name,int n,std::initializer_list<void*> parameters,
                std::initializer_list<int> info) {
  std::vector<void*> p(parameters); std::vector<int> i(info);
  launch_iris(device,name,((size_t(n)+255)/256)*256,256,p.size(),p.data(),i.data());
}
void suffixArray(Buffer&,Buffer&,int,int);


#ifdef DEBUG
void print_suffix(const char *cc, int i)
{
  printf("%d: ", i);
  for (unsigned j = i; j < strlen(cc); j++)
    printf("%c", cc[j]);
  printf("\n");
}
#endif

bool read_data(char *filename, char *buffer, int num) {
  FILE *fh;
  fh = fopen(filename, "r");
  if (fh == NULL) {
    printf("Failed to open file %s\n", filename);
    return true;
  }
  fread(buffer, 1, num, fh);
  buffer[num] = '\0';
  fclose(fh);
  return false;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dataset> <dataset size> <repeat>\n", argv[0]);
    printf("size of dataset to evaluate (10 - 1000000)\n");
    return 1;
  }

  char* filename = argv[1];

#ifdef DEBUG
  const int n = 100;
#else
  const int n = atoi(argv[2]);
#endif

  const int repeat = atoi(argv[3]);

  //load data buffer with a genome file
  char *data = (char *) malloc ((n + 1)*sizeof(char));
  bool fail = read_data(filename, data, n);
  if (data == NULL || fail) {
    if (data) free(data);
    return 1;
  }

  iris::Platform platform;
  device=init_iris(platform,argc,argv);
  Buffer input(n+3),result(n+3);
  auto& h_inp=input.host;
  auto& h_SA=result.host;

  int i;
  for (i = 0; i < n; i++) h_inp[i] = (int)data[i];
  h_inp[i] = 0; h_inp[i + 1] = 0; h_inp[i + 2] = 0; //prepare for triples

  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    input.memory.update(h_inp.data());
    result.memory.update(h_SA.data());
    suffixArray(input,result,n,MAX_ALPHA);
    flush_iris(device,result.memory);
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Average suffix array construct time (input size = %d): %f (s)\n", 
         n, time.count() / repeat);

  long sum = 0;
  for (i = 0; i < n/2; i++)
    sum += (h_SA[2*i] + h_SA[2*i+1]) / abs(h_SA[2*i] - h_SA[2*i+1]);
  if (n % 2) sum -= h_SA[n-1];
  printf("checksum = 0x%lx\n", sum);
  
#ifdef DEBUG
  //peek sorted suffixes from data set
  for(i = 0; i < 8; i++) {
    printf("No.%d Index.", i); 
    print_suffix(data, h_SA[i]);
  }

  printf("...\n...\n");

  for(i = n-8; i < n; i++) {
    printf("No.%d Index.", i); 
    print_suffix(data, h_SA[i]);
  }
#endif

  if(const char* path=getenv("SA_VALIDATION_DUMP")) {
    FILE* fp=fopen(path,"wb"); if(!fp) return 1;
    size_t count=fwrite(h_SA.data(),sizeof(int),n,fp);
    fclose(fp); if(count!=size_t(n)) return 1;
  }
  free(data);
  const int errors=platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}

void copy_buffer(Buffer& src,Buffer& dst,int n) {
  run_kernel("copy_int",n,{&src.memory,&dst.memory,&n},{iris_r,iris_rw,sizeof(n)});
}
void stable_sort(Buffer& keys,Buffer& values,int n) {
  int capacity=1;while(capacity<n) capacity*=2;
  Buffer work_keys(capacity),work_values(capacity),order(capacity);
  run_kernel("sort_init",capacity,{&keys.memory,&values.memory,&work_keys.memory,&work_values.memory,&order.memory,&n,&capacity},
       {iris_r,iris_r,iris_w,iris_w,iris_w,sizeof(n),sizeof(capacity)});
  for(int width=2;width<=capacity;width*=2) for(int stride=width/2;stride;stride/=2)
    run_kernel("sort_step",capacity,{&work_keys.memory,&work_values.memory,&order.memory,&capacity,&width,&stride},
       {iris_rw,iris_rw,iris_rw,sizeof(capacity),sizeof(width),sizeof(stride)});
  run_kernel("sort_finish",n,{&work_keys.memory,&work_values.memory,&keys.memory,&values.memory,&n},
       {iris_r,iris_r,iris_rw,iris_rw,sizeof(n)});
}
void exclusive_scan(Buffer& data,int n) {
  int blocks=(n+255)/256;
  Buffer sums(blocks);
  run_kernel("scan_blocks",n,{&data.memory,&sums.memory,&n},{iris_rw,iris_w,sizeof(n)});
  run_kernel("scan_sums",1,{&sums.memory,&blocks},{iris_rw,sizeof(blocks)});
  run_kernel("scan_add",n,{&data.memory,&sums.memory,&n},{iris_rw,iris_r,sizeof(n)});
}
void suffixArray(Buffer& s,Buffer& SA,int n,int K) {
  (void)K;
  int n0=(n+2)/3,n2=n/3,n02=n0+n2;
  Buffer s12(n02+3),SA12(n02+3),s0(n0),SA0(n0),scan(n02+3);
  run_kernel("Init_d_s12",n02,{&s12.memory,&n02},{iris_rw,sizeof(n02)});
  for(int i=2;i>=0;--i) {
    run_kernel("keybits",n02,{&SA12.memory,&s12.memory,&s.memory,&n02,&i},
                 {iris_rw,iris_r,iris_r,sizeof(n02),sizeof(i)});
    stable_sort(SA12,s12,n02);
  }
  copy_buffer(s12,SA12,n02+3);
  run_kernel("InitScan",n02,{&s.memory,&SA12.memory,&scan.memory,&n02},
           {iris_r,iris_r,iris_rw,sizeof(n02)});
  exclusive_scan(scan,n02+1);
  run_kernel("Set_suffix_rank",n02,{&s12.memory,&SA12.memory,&scan.memory,&n02,&n0},
           {iris_rw,iris_r,iris_r,sizeof(n02),sizeof(n0)});
  flush_iris(device,scan.memory);
  int max_rank=scan.host[n02];
  if(max_rank<n02) {
    suffixArray(s12,SA12,n02,max_rank);
    run_kernel("Store_unique_ranks",n02,{&s12.memory,&SA12.memory,&n02},{iris_rw,iris_r,sizeof(n02)});
  } else {
    run_kernel("Compute_SA_From_UniqueRank",n02,{&s12.memory,&SA12.memory,&n02},{iris_r,iris_rw,sizeof(n02)});
  }
  run_kernel("InitScan2",n02,{&SA12.memory,&scan.memory,&n0,&n02},{iris_r,iris_rw,sizeof(n0),sizeof(n02)});
  exclusive_scan(scan,n02);
  run_kernel("Set_S0",n02,{&s0.memory,&SA12.memory,&scan.memory,&n0,&n02},
           {iris_rw,iris_r,iris_r,sizeof(n0),sizeof(n02)});
  int zero=0;
  run_kernel("keybits",n0,{&SA0.memory,&s0.memory,&s.memory,&n0,&zero},
           {iris_rw,iris_r,iris_r,sizeof(n0),sizeof(zero)});
  stable_sort(SA0,s0,n0);
  copy_buffer(s0,SA0,n0);
  run_kernel("merge_suffixes",n0+n02,{&SA0.memory,&SA12.memory,&SA.memory,&s.memory,&s12.memory,&n0,&n02,&n},
       {iris_r,iris_r,iris_rw,iris_r,iris_r,sizeof(n0),sizeof(n02),sizeof(n)});
}
