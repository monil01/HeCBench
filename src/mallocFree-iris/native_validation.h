#pragma once
// Default-preserving reference diagnostics shared by the CUDA/HIP sources.
// Verification is opt-in; allocation-kind-correct deallocation is unconditional.
#ifdef __HIPCC__
#define HB_CHECK(expression) do { hipError_t e=(expression); if(e!=hipSuccess) { fprintf(stderr,"HIP allocation validation failed: %s\n",hipGetErrorString(e)); exit(2); } } while(0)
static void native_release(int* p, int kind) { if(kind==2) HB_CHECK(hipHostFree(p)); else HB_CHECK(hipFree(p)); }
#else
#define HB_CHECK(expression) do { cudaError_t e=(expression); if(e!=cudaSuccess) { fprintf(stderr,"CUDA allocation validation failed: %s\n",cudaGetErrorString(e)); exit(2); } } while(0)
static void native_release(int* p, int kind) { if(kind==2) HB_CHECK(cudaFreeHost(p)); else HB_CHECK(cudaFree(p)); }
#endif
#ifdef HECBENCH_MALLOC_VERIFY
__global__ void hecbench_allocation_pattern_native(int* data, size_t n, int tag) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) data[i]=static_cast<int>(i%251)+tag*257;
}
static void native_validate_allocations(int** pointers, size_t bytes, int count, int kind) {
  const size_t n=bytes/sizeof(int);
  for(int j=0;j<count;++j) {
#ifdef __HIPCC__
    hipPointerAttribute_t attributes;
    HB_CHECK(hipPointerGetAttributes(&attributes,pointers[j]));
    if((kind==1 && !attributes.isManaged) ||
       (kind==0 && (attributes.type!=hipMemoryTypeDevice || attributes.isManaged)) ||
       (kind==2 && attributes.type!=hipMemoryTypeHost)) { fprintf(stderr,"FAIL native allocation kind\n"); exit(1); }
    hipLaunchKernelGGL(hecbench_allocation_pattern_native, dim3((n+255)/256), dim3(256), 0, 0, pointers[j], n, j);
    HB_CHECK(hipGetLastError());
#else
    hecbench_allocation_pattern_native<<<(n+255)/256,256>>>(pointers[j],n,j);
    HB_CHECK(cudaGetLastError());
#endif
  }
  int* output=static_cast<int*>(malloc(bytes));
  if(!output) exit(2);
  for(int j=0;j<count;++j) {
#ifdef __HIPCC__
    HB_CHECK(hipMemcpy(output,pointers[j],bytes,hipMemcpyDefault));
#else
    HB_CHECK(cudaMemcpy(output,pointers[j],bytes,cudaMemcpyDefault));
#endif
    for(size_t i=0;i<n;++i) if(output[i]!=static_cast<int>(i%251)+j*257) {
      fprintf(stderr,"FAIL allocation kind=%d buffer=%d element=%zu\n",kind,j,i); exit(1);
    }
  }
  free(output);
  printf("PASS allocation kind=%d bytes=%zu buffers=%d; every element checked\n",kind,bytes,count);
}
#else
static void native_validate_allocations(int**, size_t, int, int) {}
#endif
