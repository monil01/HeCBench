#include <cuda_runtime.h>
#include <climits>
typedef unsigned int uint;
#include "QuicksortKernels.cuh"

extern "C" __global__ void gqsort_uint(unsigned int* d, unsigned int* dn, block_record<unsigned int>* blocks, parent_record* parents, work_record<unsigned int>* result) { gqsort_kernel(d,dn,blocks,parents,result); }
extern "C" __global__ void lqsort_uint(unsigned int* d, unsigned int* dn, work_record<unsigned int>* seqs) { lqsort_kernel(d,dn,seqs); }

extern "C" __global__ void gqsort_float(float* d, float* dn, block_record<float>* blocks, parent_record* parents, work_record<float>* result) { gqsort_kernel(d,dn,blocks,parents,result); }
extern "C" __global__ void lqsort_float(float* d, float* dn, work_record<float>* seqs) { lqsort_kernel(d,dn,seqs); }

extern "C" __global__ void gqsort_double(double* d, double* dn, block_record<double>* blocks, parent_record* parents, work_record<double>* result) { gqsort_kernel(d,dn,blocks,parents,result); }
extern "C" __global__ void lqsort_double(double* d, double* dn, work_record<double>* seqs) { lqsort_kernel(d,dn,seqs); }

