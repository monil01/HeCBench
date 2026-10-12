#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define NUM_ELEMENTS 4096
#define COUNT 4096

typedef struct {
  unsigned int s0;
  unsigned int s1;
  unsigned int s2;
  unsigned int s3;
  unsigned int s4;
  unsigned int s5;
  unsigned int s6;
  unsigned int s7;
  unsigned int s8;
  unsigned int s9;
  unsigned int sa;
  unsigned int sb;
  unsigned int sc;
  unsigned int sd;
  unsigned int se;
  unsigned int sf;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct {
  ARRAY_MEMBER_T s0;
  ARRAY_MEMBER_T s1;
  ARRAY_MEMBER_T s2;
  ARRAY_MEMBER_T s3;
  ARRAY_MEMBER_T s4;
  ARRAY_MEMBER_T s5;
  ARRAY_MEMBER_T s6;
  ARRAY_MEMBER_T s7;
  ARRAY_MEMBER_T s8;
  ARRAY_MEMBER_T s9;
  ARRAY_MEMBER_T sa;
  ARRAY_MEMBER_T sb;
  ARRAY_MEMBER_T sc;
  ARRAY_MEMBER_T sd;
  ARRAY_MEMBER_T se;
  ARRAY_MEMBER_T sf;
} NON_INTERLEAVED_T;

#include "util.h"

static void add_test_interleaved(INTERLEAVED_T* h_dst, const INTERLEAVED_T* h_src,
                                 int repeat, unsigned int num_elements)
{
  const size_t num_bytes = sizeof(INTERLEAVED_T) * num_elements;
  iris::DMem mem_dst(h_dst, num_bytes);
  iris::DMem mem_src(const_cast<INTERLEAVED_T*>(h_src), num_bytes);
  const size_t gws[1] = { static_cast<size_t>(((num_elements + 255U) / 256U) * 256U) };
  const size_t lws[1] = { 256 };

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[3] = { &mem_dst, &mem_src, const_cast<unsigned int*>(&num_elements) };
    int pinfo[3] = { iris_rw, iris_r, sizeof(num_elements) };
    task.kernel("add_kernel_interleaved", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    if (n == repeat - 1) task.flush_out(mem_dst);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel (interleaved) execution time %f (s)\n",
              (time * 1e-9f) / repeat);
}

static void add_test_non_interleaved(NON_INTERLEAVED_T* h_dst,
                                     const NON_INTERLEAVED_T* h_src,
                                     int repeat, unsigned int num_elements)
{
  const size_t num_bytes = sizeof(NON_INTERLEAVED_T);
  iris::DMem mem_dst(h_dst, num_bytes);
  iris::DMem mem_src(const_cast<NON_INTERLEAVED_T*>(h_src), num_bytes);
  const size_t gws[1] = { static_cast<size_t>(((num_elements + 255U) / 256U) * 256U) };
  const size_t lws[1] = { 256 };

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[3] = { &mem_dst, &mem_src, const_cast<unsigned int*>(&num_elements) };
    int pinfo[3] = { iris_rw, iris_r, sizeof(num_elements) };
    task.kernel("add_kernel_non_interleaved", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    if (n == repeat - 1) task.flush_out(mem_dst);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel (non-interleaved) execution time %f (s)\n",
              (time * 1e-9f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst;
  INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst;
  initialize(interleaved_src, interleaved_dst, non_interleaved_src,
             non_interleaved_dst, NUM_ELEMENTS);

  add_test_non_interleaved(&non_interleaved_dst, &non_interleaved_src, repeat,
                           NUM_ELEMENTS);
  add_test_interleaved(interleaved_dst, interleaved_src, repeat, NUM_ELEMENTS);
  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);
  std::printf("PASS\n");

  platform.finalize();
  return 0;
}
