// Modified for IRIS DMem backend/host conversion, 2026-10-13; see LICENSE.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "iris_support.h"

#include "reference.h"

static size_t bounded_size(const char* name,size_t fallback) {
  const char* text=std::getenv(name);
  size_t value=text?std::strtoull(text,nullptr,10):fallback;
  if(value<1) throw std::runtime_error("Shape must be positive");
  return value;
}
static void dump_ints(const char* name,const int* values,size_t count) {
  if(!std::getenv("HECBENCH_DUMP")) return;
  FILE* fp=fopen(name,"wb");
  if(!fp || fwrite(values,sizeof(int),count,fp)!=count)
    throw std::runtime_error("Artifact write failed");
  fclose(fp);
}
int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if(repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();

  const float threshold = 0.4f;
  const int classwise_topK = 10;
  const size_t num_classes = bounded_size("HECBENCH_SCORE_CLASSES",1000);
  const size_t num_priors = 4096;
  const int batch_size = bounded_size("HECBENCH_SCORE_BATCH",128);
  const int block_size = 256; 
  
  size_t lws[]={static_cast<size_t>(block_size),1};
  size_t gws[]={num_classes*block_size,static_cast<size_t>(batch_size)};

  size_t indices_size = batch_size * num_classes * classwise_topK;
  size_t indices_size_bytes = indices_size * sizeof(int);

  size_t count_size = batch_size * num_classes;
  size_t count_size_bytes = count_size * sizeof(int);

  size_t scores_size = batch_size * num_classes * num_priors;
  size_t scores_size_bytes = scores_size * sizeof(int);

  float *scores = (float*) malloc (scores_size_bytes);
  int *count = (int*) malloc (count_size_bytes);
  int *count_ref = (int*) malloc (count_size_bytes);
  memset(count_ref, 0, count_size_bytes);
  int *indices = (int*) malloc (indices_size_bytes);
  int *indices_ref = (int*) malloc (indices_size_bytes);

  memset(indices,0,indices_size_bytes);
  iris::DMem d_scores(scores,scores_size_bytes);
  iris::DMem d_count(count,count_size_bytes);
  iris::DMem d_indices(indices,indices_size_bytes);

  srand(123);
  for (int b = 0; b < batch_size; b++) {
    for (size_t c = 0; c < num_classes; c++) {
      float *s = scores + b * num_classes * num_priors + c * num_priors;
      for (size_t p = 0; p < num_priors; p++)
        s[p] = p * 1.0 / num_priors;
      for (int i = num_priors-1; i > 0; i--)
        std::swap(s[i], s[rand() % (i+1)]);
    }
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* args[]={&d_indices,&d_count,&d_scores,(void*)&threshold,(void*)&classwise_topK,(void*)&num_classes,(void*)&num_priors};
    int info[]={iris_rw,iris_w,iris_r,sizeof(threshold),sizeof(classwise_topK),sizeof(num_classes),sizeof(num_priors)};
    iris_check(task.kernel("findTopK",2,nullptr,gws,lws,7,args,info),"kernel");
    iris_check(task.submit(device,nullptr,true),"HIP task");
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", time * 1e-6f / repeat);

  iris::Task flush;
  iris_check(flush.flush_out(d_indices),"flush indices");
  iris_check(flush.flush_out(d_count),"flush counts");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");

  reference<float, 2048>(indices_ref, count_ref, scores, threshold, classwise_topK, batch_size, num_classes, num_priors);

  unsigned checksum = 0; 
  for (int b = 0; b < batch_size; b++) {
    for (size_t c = 0; c < num_classes; c++) {
      size_t offset = b * num_classes * classwise_topK + c * classwise_topK;
      auto topK = indices + offset; 
      auto topK_ref = indices_ref + offset;
      std::sort(topK, topK+classwise_topK);
      std::sort(topK_ref, topK_ref+classwise_topK);
      checksum += memcmp(topK, topK_ref, sizeof(int)*classwise_topK);
    }
  }
  checksum += memcmp(count, count_ref, count_size_bytes);
  printf("%s\n", checksum == 0 ? "PASS" : "FAIL");

  dump_ints("score_indices.bin",indices,indices_size);
  dump_ints("score_count.bin",count,count_size);
  dump_ints("score_indices_ref.bin",indices_ref,indices_size);
  dump_ints("score_count_ref.bin",count_ref,count_size);

  free(indices);
  free(indices_ref);
  free(count);
  free(count_ref);
  free(scores);

  iris_check(platform.finalize(),"finalize");
  return checksum==0 ? 0 : 1;
}
