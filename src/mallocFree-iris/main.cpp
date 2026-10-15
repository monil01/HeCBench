/*
   Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <vector>
#include <cstdint>

#define NUM_SIZE 19  //size up to 16M
#ifndef NUM_ITER
#define NUM_ITER 500 //Total GPU memory up to 16M*500=8G
#endif

#define Clock() std::chrono::steady_clock::now()

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

void setup(size_t *size, int &num, int **pA, const size_t totalGlobalMem) {

  for (int i = 0; i < num; i++) {
    size[i] = 1 << (i + 6);
    if((NUM_ITER + 1) * size[i] > totalGlobalMem) {
      num = i;
      break;
    }
  }
  *pA = (int*)malloc(size[num - 1]);
  valSet(*pA, 1, size[num - 1]);
}


static void checked(int rc) {
  if (rc != IRIS_SUCCESS || iris_error_count()) {
    fprintf(stderr, "IRIS allocation error %d\n", rc); exit(2);
  }
}
static void allocate(iris::DMem** out, size_t bytes, int kind) {
  *out = new iris::DMem(bytes);
  checked(iris_dmem_set_hip_allocation_kind((*out)->mem(), kind));
  void* unused = nullptr;
  // Force physical allocation; a lazy DMem constructor alone is not a GPU
  // allocation benchmark. The native pointer is never used by host code.
  checked(iris_mem_arch((*out)->mem(), 0, &unused));
}
static void release(iris::DMem* memory) {
  delete memory;
  checked(IRIS_SUCCESS);
}
static void validate_allocations(iris::DMem** memory, size_t bytes, int count, int kind) {
  const size_t n = bytes / sizeof(int);
  size_t local[] = {256}, global[] = {((n+255)/256)*256};
  for (int j = 0; j < count; ++j) {
    void* pointer = nullptr;
    checked(iris_mem_arch(memory[j]->mem(), 0, &pointer));
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess ||
        (kind == 1 && !attributes.isManaged) ||
        (kind == 0 && (attributes.type != hipMemoryTypeDevice || attributes.isManaged)) ||
        (kind == 2 && attributes.type != hipMemoryTypeHost)) {
      fprintf(stderr, "FAIL physical allocation kind %d\n", kind); exit(1);
    }
    iris::Task task;
    void* args[] = {memory[j], const_cast<size_t*>(&n), &j};
    int info[] = {iris_w, sizeof(n), sizeof(j)};
    checked(task.kernel("allocation_pattern", 1, nullptr, global, local, 3, args, info));
    checked(task.submit(0, nullptr, true));
  }
  std::vector<int> output(n);
  for (int j = 0; j < count; ++j) {
    checked(memory[j]->fetch(output.data()));
    for (size_t i = 0; i < n; ++i) {
      const int expected = static_cast<int>(i % 251) + j*257;
      if (output[i] != expected) {
        fprintf(stderr, "FAIL allocation kind=%d buffer=%d element=%zu: %d != %d\n",
                kind, j, i, output[i], expected); exit(1);
      }
    }
  }
  printf("PASS allocation kind=%d bytes=%zu buffers=%d; every element checked\n", kind, bytes, count);
}

void testInit(size_t size, int type) {

  printf("Initial allocation and deallocation\n");

  iris::DMem *Ad = nullptr;
  auto start = Clock();
  if (type == 0)
    allocate(&Ad, size, 1);
  else if (type == 1)
    allocate(&Ad, size, 0);
  else if (type == 2)
    allocate(&Ad, size, 2);

  auto end = Clock();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  if (type == 0)
    printf("IRIS DMem managed(%zu) takes %lf us\n", size, time * 1e-3);
  else if (type == 1)
    printf("IRIS DMem device(%zu) takes %lf us\n", size, time * 1e-3);
  else if (type == 2)
    printf("IRIS DMem coherent mapped host(%zu) takes %lf us\n", size, time * 1e-3);

  start = Clock();
  release(Ad);
  end = Clock();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("IRIS DMem release(%zu) takes %lf us\n", size, time * 1e-3);
  printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <total global memory size in bytes>\n", argv[0]);
    return 1;
  }

  const size_t totalGlobalMem = atol(argv[1]);
  if (totalGlobalMem < static_cast<size_t>(NUM_ITER+1)*64) return 1;
  iris::Platform platform;
  checked(platform.init(&argc, &argv, 1));

  size_t size[NUM_SIZE] = { 0 };
  iris::DMem *Ad[NUM_ITER] = { nullptr };

  int num = NUM_SIZE;
  int *A;
  setup(size, num, &A, totalGlobalMem);

  printf("\n==== Evaluate IRIS DMem managed allocation/release ====\n");
  testInit(size[0], 0);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      allocate(&Ad[j], size[i], 1);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem managed(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    validate_allocations(Ad, size[i], NUM_ITER, 1);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      release(Ad[j]);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem release(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  printf("\n==== Evaluate IRIS DMem device allocation/release ====\n");
  testInit(size[0], 1);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      allocate(&Ad[j], size[i], 0);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem device(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    validate_allocations(Ad, size[i], NUM_ITER, 0);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      release(Ad[j]);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem release(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  printf("\n==== Evaluate IRIS DMem coherent mapped host allocation/release ====\n");
  testInit(size[0], 2);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      allocate(&Ad[j], size[i], 2);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem coherent mapped host(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    validate_allocations(Ad, size[i], NUM_ITER, 2);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      release(Ad[j]);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("IRIS DMem release(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  free(A);
  checked(platform.finalize());
  return iris_error_count() ? 1 : 0;
}
