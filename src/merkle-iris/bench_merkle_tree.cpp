#include <random>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include "bench_merkle_tree.hpp"
#include "rescue_prime.hpp"
#include "iris_host.h"

uint64_t benchmark_merklize_approach_1(const size_t leaf_count,const size_t wg_size) {
 const size_t leaves_size=sizeof(ulong)*leaf_count*DIGEST_SIZE;
 const size_t mds_size=sizeof(ulong4)*STATE_WIDTH*3,ark_size=sizeof(ulong4)*NUM_ROUNDS*3;
 ulong* leaves_h=(ulong*)aligned_alloc(32,leaves_size);
 ulong4* mds_h=(ulong4*)aligned_alloc(32,mds_size);
 ulong4* ark1_h=(ulong4*)aligned_alloc(32,ark_size),*ark2_h=(ulong4*)aligned_alloc(32,ark_size);
 if(!leaves_h || !mds_h || !ark1_h || !ark2_h) std::exit(1);
 {
  std::mt19937 gen(19937);std::uniform_int_distribution<uint64_t> dis(1ul,MOD);
  for(uint64_t i=0;i<leaf_count*DIGEST_SIZE;++i) leaves_h[i]=static_cast<ulong>(dis(gen));
 }
 prepare_mds(mds_h);prepare_ark1(ark1_h);prepare_ark2(ark2_h);
 uint64_t ts=0;
 {
  iris::DMem leaves_d(leaves_h,leaves_size),mds_d(mds_h,mds_size),ark1_d(ark1_h,ark_size),ark2_d(ark2_h,ark_size);
#ifdef DEBUG
  ulong* intermediates_h=(ulong*)aligned_alloc(32,leaves_size);
  iris::DMem intermediates_d(intermediates_h,leaves_size);
#else
  iris::DMem intermediates_d(leaves_size);
#endif
  assert((leaf_count & (leaf_count-1))==0);assert(wg_size <= (leaf_count>>1));
  auto start_time=std::chrono::high_resolution_clock::now();
  const size_t output_offset=leaf_count>>1;
  launch("iris_MerklizeRescuePrimeApproach1Phase0",Dim(output_offset/wg_size),Dim(wg_size),output_offset,std::as_const(leaves_d),intermediates_d,std::as_const(mds_d),std::as_const(ark1_d),std::as_const(ark2_d));
  const size_t rounds=static_cast<size_t>(log2(static_cast<double>(leaf_count>>1)));
  for(size_t r=0;r<rounds;++r) {
   const size_t offset=leaf_count>>(r+2);
   int block_size=offset<wg_size ? offset:wg_size;
   launch("iris_MerklizeRescuePrimeApproach1Phase1",Dim(offset/block_size),Dim(block_size),offset,intermediates_d,std::as_const(mds_d),std::as_const(ark1_d),std::as_const(ark2_d));
  }
  auto end_time=std::chrono::high_resolution_clock::now();
  ts=std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count();
#ifdef DEBUG
  flush(intermediates_d);
  for(int i=0;i<leaf_count*DIGEST_SIZE;++i) printf("%d %lu\n",i,intermediates_h[i]);
  free(intermediates_h);
#endif
 }
 free(leaves_h);free(mds_h);free(ark1_h);free(ark2_h);
 return ts;
}
