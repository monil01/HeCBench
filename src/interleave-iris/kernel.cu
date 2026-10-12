#define COUNT 4096

typedef struct {
  unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
  unsigned int s8, s9, sa, sb, sc, sd, se, sf;
} INTERLEAVED_T;

#define NUM_ELEMENTS 4096
typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct {
  ARRAY_MEMBER_T s0, s1, s2, s3, s4, s5, s6, s7;
  ARRAY_MEMBER_T s8, s9, sa, sb, sc, sd, se, sf;
} NON_INTERLEAVED_T;

extern "C" __global__ void add_kernel_interleaved(
    INTERLEAVED_T* dest_ptr, const INTERLEAVED_T* src_ptr,
    const unsigned int num_elements)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_elements) {
    for (unsigned int i = 0; i < COUNT; i++) {
      dest_ptr[tid].s0 += src_ptr[tid].s0;
      dest_ptr[tid].s1 += src_ptr[tid].s1;
      dest_ptr[tid].s2 += src_ptr[tid].s2;
      dest_ptr[tid].s3 += src_ptr[tid].s3;
      dest_ptr[tid].s4 += src_ptr[tid].s4;
      dest_ptr[tid].s5 += src_ptr[tid].s5;
      dest_ptr[tid].s6 += src_ptr[tid].s6;
      dest_ptr[tid].s7 += src_ptr[tid].s7;
      dest_ptr[tid].s8 += src_ptr[tid].s8;
      dest_ptr[tid].s9 += src_ptr[tid].s9;
      dest_ptr[tid].sa += src_ptr[tid].sa;
      dest_ptr[tid].sb += src_ptr[tid].sb;
      dest_ptr[tid].sc += src_ptr[tid].sc;
      dest_ptr[tid].sd += src_ptr[tid].sd;
      dest_ptr[tid].se += src_ptr[tid].se;
      dest_ptr[tid].sf += src_ptr[tid].sf;
    }
  }
}

extern "C" __global__ void add_kernel_non_interleaved(
    NON_INTERLEAVED_T* dest_ptr, const NON_INTERLEAVED_T* src_ptr,
    const unsigned int num_elements)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_elements) {
    for (unsigned int i = 0; i < COUNT; i++) {
      dest_ptr->s0[tid] += src_ptr->s0[tid];
      dest_ptr->s1[tid] += src_ptr->s1[tid];
      dest_ptr->s2[tid] += src_ptr->s2[tid];
      dest_ptr->s3[tid] += src_ptr->s3[tid];
      dest_ptr->s4[tid] += src_ptr->s4[tid];
      dest_ptr->s5[tid] += src_ptr->s5[tid];
      dest_ptr->s6[tid] += src_ptr->s6[tid];
      dest_ptr->s7[tid] += src_ptr->s7[tid];
      dest_ptr->s8[tid] += src_ptr->s8[tid];
      dest_ptr->s9[tid] += src_ptr->s9[tid];
      dest_ptr->sa[tid] += src_ptr->sa[tid];
      dest_ptr->sb[tid] += src_ptr->sb[tid];
      dest_ptr->sc[tid] += src_ptr->sc[tid];
      dest_ptr->sd[tid] += src_ptr->sd[tid];
      dest_ptr->se[tid] += src_ptr->se[tid];
      dest_ptr->sf[tid] += src_ptr->sf[tid];
    }
  }
}
