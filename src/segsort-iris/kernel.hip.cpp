#include <hip/hip_runtime.h>
#include <cstdint>
#define SHFL_XOR(b,c) __shfl_xor(b,c,32)
#define SHFL_UP(b,c) __shfl_up(b,c,32)
#define SHFL(b,c) __shfl(b,c,32)
#include "device/bb_bin.cuh"
#include "device/bb_comput_s.cuh"
#include "device/bb_comput_l.cuh"
#include "device/bb_comput_s_keys.cuh"
#include "device/bb_comput_l_keys.cuh"
extern "C" __global__ void bin_histo(int *counter,const int *seg,int n){bb_bin_histo(counter,seg,seg+1,n);}
extern "C" __global__ void bin_group(int *bin,int *counter,const int *seg,int n){bb_bin_group(bin,counter,seg,seg+1,n);}
extern "C" __global__ void keys_bin0(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_copy(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin1(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp2_tc1_r2_r2_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin2(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp2_tc2_r3_r4_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin3(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp2_tc4_r5_r8_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin4(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp4_tc4_r9_r16_strd(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin5(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp8_tc4_r17_r32_strd(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin6(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp16_tc4_r33_r64_strd(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin7(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp8_tc16_r65_r128_strd(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin8(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp32_tc8_r129_r256_strd(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin9(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_tc4_r257_r512_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin10(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_tc4_r513_r1024_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_bin11(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk512_tc4_r1025_r2048_orig(key,keyB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void keys_long_sort(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_block_sort(key,keyB,seg,seg+1,bin+offset,2048);}
extern "C" __global__ void keys_long_merge(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_block_merge(key,keyB,seg,seg+1,bin+offset,stride,2048);}
extern "C" __global__ void keys_long_copy(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_copy(key,keyB,seg,seg+1,bin+offset,2048);}
extern "C" __global__ void pairs_bin0(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_copy(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin1(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp2_tc1_r2_r2_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin2(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp2_tc2_r3_r4_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin3(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp2_tc4_r5_r8_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin4(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp4_tc4_r9_r16_strd(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin5(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp8_tc4_r17_r32_strd(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin6(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_wp16_tc4_r33_r64_strd(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin7(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp8_tc16_r65_r128_strd(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin8(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_wp32_tc8_r129_r256_strd(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin9(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk128_tc4_r257_r512_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin10(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk256_tc4_r513_r1024_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_bin11(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n){gen_bk512_tc4_r1025_r2048_orig(key,val,keyB,valB,seg,seg+1,bin+offset,n);}
extern "C" __global__ void pairs_long_sort(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_block_sort(key,val,keyB,valB,seg,seg+1,bin+offset,2048);}
extern "C" __global__ void pairs_long_merge(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_block_merge(key,val,keyB,valB,seg,seg+1,bin+offset,stride,2048);}
extern "C" __global__ void pairs_long_copy(int *key,uint64_t *val,int *keyB,uint64_t *valB,const int *seg,const int *bin,int offset,int n,int stride){kern_copy(key,val,keyB,valB,seg,seg+1,bin+offset,2048);}
