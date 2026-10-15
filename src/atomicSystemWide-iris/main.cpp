/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A program demonstrating use of system-wide atomics on migratable memory.
 */
 
#include <iris/iris_runtime.h>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)
#define IRIS_CHECK(call) do { int e = (call); if (e != IRIS_SUCCESS) { \
  fprintf(stderr, "%s: IRIS error %d\n", #call, e); return 2; } } while (0)

void atomicKernel_CPU(int *atom_arr, int no_of_threads, const int loop_num)
{

  for (int i=no_of_threads; i<2*no_of_threads; i++)
  {
    for (int j=0; j < loop_num; j++)
    {
      // Atomic addition
      __sync_fetch_and_add(&atom_arr[0],10);

      // Atomic exchange
      __sync_lock_test_and_set(&atom_arr[1], i);

      // Atomic maximum
      int old, expected;
      do {
        expected = __sync_val_compare_and_swap(&atom_arr[2], 0, 0);
        old = __sync_val_compare_and_swap(&atom_arr[2], expected, max(expected, i));
      } while (old != expected);

      // Atomic minimum
      do {
        expected = __sync_val_compare_and_swap(&atom_arr[3], 0, 0);
        old = __sync_val_compare_and_swap(&atom_arr[3], expected, min(expected, i));
      } while (old != expected);

      // Atomic increment (modulo 17+1)
      //int limit = 17;
      //do {
      //  expected = atom_arr[4];
      //  old = __sync_val_compare_and_swap(&atom_arr[4], expected, (expected >= limit) ? 0 : expected+1);
      //} while (old != expected);

      // Atomic decrement
      //limit = 137;
      //do {
      //  expected = atom_arr[5];
      //  old = __sync_val_compare_and_swap(&atom_arr[5], expected, ((expected == 0) || (expected > limit)) ? limit : expected-1);
      //} while (old != expected);

      // Atomic compare-and-swap
      __sync_val_compare_and_swap(&atom_arr[6], i-1, i);

      // Bitwise atomic instructions

      // Atomic AND
      __sync_fetch_and_and(&atom_arr[7] , 2 * i + 7);

      // Atomic OR
      __sync_fetch_and_or(&atom_arr[8], static_cast<int>(1u << (static_cast<unsigned>(i) & 31u)));

      // Atomic XOR
      // 11th element should be 0xff
      __sync_fetch_and_xor(&atom_arr[9] ,i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int verify(int *testData, const int len, const int loop_num)
{
  int val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val += 10;
  }

  if (val != testData[0])
  {
    printf("atomicAdd failed val = %d testData = %d\n", val, testData[0]);
    return false;
  }

  val = 0;

  bool found = false;

  for (int i = 0; i < len; ++i)
  {
    // second element should be a member of [0, len)
    if (i == testData[1])
    {
      found = true;
      break;
    }
  }

  if (!found)
  {
    printf("atomicExch failed\n");
    return false;
  }

  val = -(1 << 8);

  for (int i = 0; i < len; ++i)
  {
    // third element should be len-1
    val = max(val, i);
  }

  if (val != testData[2])
  {
    printf("atomicMax failed\n");
    return false;
  }

  val = 1 << 8;

  for (int i = 0; i < len; ++i)
  {
    val = min(val, i);
  }

  if (val != testData[3])
  {
    printf("atomicMin failed\n");
    return false;
  }

  /*
  int limit = 17;
  val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val = (val >= limit) ? 0 : val+1;
  }

  if (val != testData[4])
  {
    printf("atomicInc failed\n");
    return false;
  }

  limit = 137;
  val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val = ((val == 0) || (val > limit)) ? limit : val-1;
  }

  if (val != testData[5])
  {
    printf("atomicDec failed\n");
    return false;
  }
  */

  found = false;

  for (int i = 0; i < len; ++i)
  {
    // seventh element should be a member of [0, len)
    if (i == testData[6])
    {
      found = true;
      break;
    }
  }

  if (!found)
  {
    printf("atomicCAS failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i)
  {
    // 8th element should be 1
    val &= (2 * i + 7);
  }

  if (val != testData[7])
  {
    printf("atomicAnd failed\n");
    return false;
  }

  val = 0;

  for (int i = 0; i < len; ++i)
  {
    // 9th element should be 0xff
    val |= (static_cast<int>(1u << (static_cast<unsigned>(i) & 31u)));
  }

  if (val != testData[8])
  {
    printf("atomicOr failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i)
  {
    // 11th element should be 0xff
    val ^= i;
  }

  if (val != testData[9])
  {
    printf("atomicXor failed\n");
    return false;
  }

  return true;
}


int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  if (argc != 2) {
    printf("Usage: %s <loop count within the kernel>\n", argv[0]);
    return 1;
  }
  int loop_num = atoi(argv[1]);
  if (loop_num <= 0 || loop_num > INT_MAX / 327680) return 2;
  const bool proof = getenv("HECBENCH_ATOMIC_PROOF") != nullptr;
  IRIS_CHECK(iris_init(&argc, &argv, 1));
  const int device = 0;
  iris_mem counters, progress_mem;
  IRIS_CHECK(iris_data_mem_create(&counters, nullptr, 10 * sizeof(int)));
  IRIS_CHECK(iris_dmem_set_hip_allocation_kind(counters, 2));
  int* atom_arr = nullptr;
  IRIS_CHECK(iris_dmem_bind_hip_shared_view(counters, device, (void**)&atom_arr));
  for (int k = 0; k < 10; ++k) atom_arr[k] = 0;
  atom_arr[7] = atom_arr[9] = 255;

  // Every physical allocation remains DMem-owned; no HIP allocation, copy,
  // launch, USM ownership override, or free is performed by this host driver.
  IRIS_CHECK(iris_data_mem_create(&progress_mem, nullptr, 80 * sizeof(int)));
  IRIS_CHECK(iris_dmem_set_hip_allocation_kind(progress_mem, 2));
  int* progress = nullptr;
  IRIS_CHECK(iris_dmem_bind_hip_shared_view(progress_mem, device, (void**)&progress));
  for (int k = 0; k < 80; ++k) progress[k] = 0;
  // A separate no-proof kernel signature avoids passing native pointer scalars.
  size_t gws[] = {16384}, lws[] = {256};
  void* args[] = {&counters, &loop_num, &progress_mem};
  int info[] = {iris_rw, sizeof(loop_num), iris_rw};
  iris_task task;
  IRIS_CHECK(iris_task_create(&task));
  IRIS_CHECK(iris_task_kernel(task, proof ? "atomicKernel" : "atomicKernelUntraced",
                             1, nullptr, gws, lws, 3, args, info));
  const auto start = std::chrono::steady_clock::now();
  IRIS_CHECK(iris_task_submit(task, device, nullptr, 0));
  if (proof) {
    const auto deadline = start + std::chrono::seconds(30);
    while (__sync_val_compare_and_swap(progress, 0, 0) == 0) {
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "GPU proof handshake timed out\n");
        return 2;
      }
    }
    __sync_lock_test_and_set(progress + 32, 1);
  }
  atomicKernel_CPU(atom_arr, 16384, loop_num);
  if (proof) __sync_lock_test_and_set(progress + 48, 1);
  IRIS_CHECK(iris_task_wait(task));
  const auto end = std::chrono::steady_clock::now();
  printf("Execution time of atomic kernels on host and device: %.9f (s)\n",
         std::chrono::duration<double>(end - start).count());
  for (int k = 0; k < 10; ++k) printf("raw[%d]=%d\n", k, atom_arr[k]);
  int ok = verify(atom_arr, 32768, loop_num);
  if (atom_arr[4] != 0 || atom_arr[5] != 0) ok = false;
  if (proof) {
    printf("proof cpu_updates=%d gpu_threads=%d gpu_updates=%d overlap=%d\n",
           16384 * loop_num, progress[16], progress[16] * loop_num, progress[64]);
    ok = ok && progress[16] == 16384 && progress[64] == 1;
  }
  IRIS_CHECK(iris_task_release(task));
  IRIS_CHECK(iris_mem_release(progress_mem));
  IRIS_CHECK(iris_mem_release(counters));
  const int errors = iris_error_count();
  IRIS_CHECK(iris_finalize());
  printf("iris_errors=%d\n", errors);
  ok = ok && errors == 0;
  printf("systemWideAtomics completed, returned %s\n", ok ? "OK" : "ERROR!");
  return ok ? 0 : 1;
}
