#pragma once
#include "operations.hpp"
namespace cuda_simd {
__device__ inline int64_t signed_lane(uint32_t u,int bits) {
  uint64_t sign=uint64_t(1)<<(bits-1);
  return (int64_t)((uint64_t(u)^sign)-sign);
}
template<int Bits,bool Signed,Op operation,bool Relu=false>
__device__ __forceinline__ uint32_t packed(uint32_t a,uint32_t b=0,uint32_t c=0,bool* hi=nullptr,bool* lo=nullptr) {
  constexpr uint32_t mask=uint32_t((uint64_t(1)<<Bits)-1);
  uint32_t result=0;uint64_t sum=0;
  #pragma unroll
  for(int lane=0;lane<32/Bits;++lane) {
    uint32_t au=(a>>(lane*Bits))&mask,bu=(b>>(lane*Bits))&mask,cu=(c>>(lane*Bits))&mask;
    int64_t x=Signed?signed_lane(au,Bits):au,y=Signed?signed_lane(bu,Bits):bu,z=Signed?signed_lane(cu,Bits):cu;
    int64_t v=0;
    if constexpr(operation==Op::Abs) v=x<0?-x:x;
    if constexpr(operation==Op::AbsDiff) v=x<y?y-x:x-y;
    if constexpr(operation==Op::AbsSat) v=x<0?-x:x;
    if constexpr(operation==Op::Add) v=x+y;
    if constexpr(operation==Op::AddSatS) v=x+y;
    if constexpr(operation==Op::AddSatU) v=x+y;
    if constexpr(operation==Op::AvgS) v=(x+y)>=0?((x+y+1)>>1):((x+y)>>1);
    if constexpr(operation==Op::AvgU) v=(x+y+1)>>1;
    if constexpr(operation==Op::MaskEq) v=(x==y)?mask:0;
    if constexpr(operation==Op::MaskGe) v=(x>=y)?mask:0;
    if constexpr(operation==Op::MaskGt) v=(x>y)?mask:0;
    if constexpr(operation==Op::MaskLe) v=(x<=y)?mask:0;
    if constexpr(operation==Op::MaskLt) v=(x<y)?mask:0;
    if constexpr(operation==Op::MaskNe) v=(x!=y)?mask:0;
    if constexpr(operation==Op::Half) v=(x+y)>>1;
    if constexpr(operation==Op::AddMax) {
      uint32_t wrapped=(au+bu)&mask;
      int64_t add=Signed?signed_lane(wrapped,Bits):wrapped;
      v=add>z?add:z;
    }
    if constexpr(operation==Op::AddMin) {
      uint32_t wrapped=(au+bu)&mask;
      int64_t add=Signed?signed_lane(wrapped,Bits):wrapped;
      v=add<z?add:z;
    }
    if constexpr(operation==Op::PredMax) v=x>y?x:y;
    if constexpr(operation==Op::PredMin) v=x<y?x:y;
    if constexpr(operation==Op::Max3) v=x>y?(x>z?x:z):(y>z?y:z);
    if constexpr(operation==Op::Max) v=x>y?x:y;
    if constexpr(operation==Op::Min3) v=x<y?(x<z?x:z):(y<z?y:z);
    if constexpr(operation==Op::Min) v=x<y?x:y;
    if constexpr(operation==Op::Neg) v=-x;
    if constexpr(operation==Op::NegSat) v=-x;
    if constexpr(operation==Op::Sad) v=x<y?y-x:x-y;
    if constexpr(operation==Op::SetEq) v=(x==y)?1:0;
    if constexpr(operation==Op::SetGe) v=(x>=y)?1:0;
    if constexpr(operation==Op::SetGt) v=(x>y)?1:0;
    if constexpr(operation==Op::SetLe) v=(x<=y)?1:0;
    if constexpr(operation==Op::SetLt) v=(x<y)?1:0;
    if constexpr(operation==Op::SetNe) v=(x!=y)?1:0;
    if constexpr(operation==Op::Sub) v=x-y;
    if constexpr(operation==Op::SubSatS) v=x-y;
    if constexpr(operation==Op::SubSatU) v=x-y;
    if constexpr(operation==Op::AbsSat||operation==Op::NegSat||operation==Op::AddSatS||operation==Op::SubSatS) {
      constexpr int64_t top=(int64_t(1)<<(Bits-1))-1,bottom=-(int64_t(1)<<(Bits-1));
      v=v>top?top:(v<bottom?bottom:v);
    }
    if constexpr(operation==Op::AddSatU||operation==Op::SubSatU) v=v>mask?mask:(v<0?0:v);
    if constexpr(Relu) v=v<0?0:v;
    if constexpr(operation==Op::PredMax||operation==Op::PredMin) {
      bool p=operation==Op::PredMax?x>=y:x<=y;
      if(lane==0&&lo)*lo=p;
      if(lane==(32/Bits-1)&&hi)*hi=p;
    }
    if constexpr(operation==Op::Sad) sum+=v;
    else result|=(uint32_t(v)&mask)<<(lane*Bits);
  }
  if constexpr(operation==Op::Sad) return uint32_t(sum);
  return result;
}
__device__ __forceinline__ uint32_t __vabs2(uint32_t a) { return packed<16,true,Op::Abs,false>(a); }
__device__ __forceinline__ uint32_t __vabs4(uint32_t a) { return packed<8,true,Op::Abs,false>(a); }
__device__ __forceinline__ uint32_t __vabsdiffs2(uint32_t a,uint32_t b) { return packed<16,true,Op::AbsDiff,false>(a,b); }
__device__ __forceinline__ uint32_t __vabsdiffs4(uint32_t a,uint32_t b) { return packed<8,true,Op::AbsDiff,false>(a,b); }
__device__ __forceinline__ uint32_t __vabsdiffu2(uint32_t a,uint32_t b) { return packed<16,false,Op::AbsDiff,false>(a,b); }
__device__ __forceinline__ uint32_t __vabsdiffu4(uint32_t a,uint32_t b) { return packed<8,false,Op::AbsDiff,false>(a,b); }
__device__ __forceinline__ uint32_t __vabsss2(uint32_t a) { return packed<16,true,Op::AbsSat,false>(a); }
__device__ __forceinline__ uint32_t __vabsss4(uint32_t a) { return packed<8,true,Op::AbsSat,false>(a); }
__device__ __forceinline__ uint32_t __vadd2(uint32_t a,uint32_t b) { return packed<16,false,Op::Add,false>(a,b); }
__device__ __forceinline__ uint32_t __vadd4(uint32_t a,uint32_t b) { return packed<8,false,Op::Add,false>(a,b); }
__device__ __forceinline__ uint32_t __vaddss2(uint32_t a,uint32_t b) { return packed<16,true,Op::AddSatS,false>(a,b); }
__device__ __forceinline__ uint32_t __vaddss4(uint32_t a,uint32_t b) { return packed<8,true,Op::AddSatS,false>(a,b); }
__device__ __forceinline__ uint32_t __vaddus2(uint32_t a,uint32_t b) { return packed<16,false,Op::AddSatU,false>(a,b); }
__device__ __forceinline__ uint32_t __vaddus4(uint32_t a,uint32_t b) { return packed<8,false,Op::AddSatU,false>(a,b); }
__device__ __forceinline__ uint32_t __vavgs2(uint32_t a,uint32_t b) { return packed<16,true,Op::AvgS,false>(a,b); }
__device__ __forceinline__ uint32_t __vavgs4(uint32_t a,uint32_t b) { return packed<8,true,Op::AvgS,false>(a,b); }
__device__ __forceinline__ uint32_t __vavgu2(uint32_t a,uint32_t b) { return packed<16,false,Op::AvgU,false>(a,b); }
__device__ __forceinline__ uint32_t __vavgu4(uint32_t a,uint32_t b) { return packed<8,false,Op::AvgU,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpeq2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskEq,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpeq4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskEq,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpges2(uint32_t a,uint32_t b) { return packed<16,true,Op::MaskGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpges4(uint32_t a,uint32_t b) { return packed<8,true,Op::MaskGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgeu2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgeu4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgts2(uint32_t a,uint32_t b) { return packed<16,true,Op::MaskGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgts4(uint32_t a,uint32_t b) { return packed<8,true,Op::MaskGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgtu2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpgtu4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmples2(uint32_t a,uint32_t b) { return packed<16,true,Op::MaskLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmples4(uint32_t a,uint32_t b) { return packed<8,true,Op::MaskLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpleu2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpleu4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmplts2(uint32_t a,uint32_t b) { return packed<16,true,Op::MaskLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmplts4(uint32_t a,uint32_t b) { return packed<8,true,Op::MaskLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpltu2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpltu4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpne2(uint32_t a,uint32_t b) { return packed<16,false,Op::MaskNe,false>(a,b); }
__device__ __forceinline__ uint32_t __vcmpne4(uint32_t a,uint32_t b) { return packed<8,false,Op::MaskNe,false>(a,b); }
__device__ __forceinline__ uint32_t __vhaddu2(uint32_t a,uint32_t b) { return packed<16,false,Op::Half,false>(a,b); }
__device__ __forceinline__ uint32_t __vhaddu4(uint32_t a,uint32_t b) { return packed<8,false,Op::Half,false>(a,b); }
__device__ __forceinline__ uint32_t __viaddmax_s16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::AddMax,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmax_s16x2_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::AddMax,true>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmax_s32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::AddMax,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmax_s32_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::AddMax,true>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmax_u16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,false,Op::AddMax,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmax_u32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,false,Op::AddMax,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_s16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::AddMin,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_s16x2_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::AddMin,true>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_s32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::AddMin,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_s32_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::AddMin,true>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_u16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,false,Op::AddMin,false>(a,b,c); }
__device__ __forceinline__ uint32_t __viaddmin_u32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,false,Op::AddMin,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vibmax_s16x2(uint32_t a,uint32_t b,bool* hi,bool* lo) { return packed<16,true,Op::PredMax,false>(a,b,0,hi,lo); }
__device__ __forceinline__ uint32_t __vibmax_s32(uint32_t a,uint32_t b,bool* p) { return packed<32,true,Op::PredMax,false>(a,b,0,p,p); }
__device__ __forceinline__ uint32_t __vibmax_u16x2(uint32_t a,uint32_t b,bool* hi,bool* lo) { return packed<16,false,Op::PredMax,false>(a,b,0,hi,lo); }
__device__ __forceinline__ uint32_t __vibmax_u32(uint32_t a,uint32_t b,bool* p) { return packed<32,false,Op::PredMax,false>(a,b,0,p,p); }
__device__ __forceinline__ uint32_t __vibmin_s16x2(uint32_t a,uint32_t b,bool* hi,bool* lo) { return packed<16,true,Op::PredMin,false>(a,b,0,hi,lo); }
__device__ __forceinline__ uint32_t __vibmin_s32(uint32_t a,uint32_t b,bool* p) { return packed<32,true,Op::PredMin,false>(a,b,0,p,p); }
__device__ __forceinline__ uint32_t __vibmin_u16x2(uint32_t a,uint32_t b,bool* hi,bool* lo) { return packed<16,false,Op::PredMin,false>(a,b,0,hi,lo); }
__device__ __forceinline__ uint32_t __vibmin_u32(uint32_t a,uint32_t b,bool* p) { return packed<32,false,Op::PredMin,false>(a,b,0,p,p); }
__device__ __forceinline__ uint32_t __vimax3_s16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::Max3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax3_s16x2_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::Max3,true>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax3_s32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::Max3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax3_s32_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::Max3,true>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax3_u16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,false,Op::Max3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax3_u32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,false,Op::Max3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimax_s16x2_relu(uint32_t a,uint32_t b) { return packed<16,true,Op::Max,true>(a,b); }
__device__ __forceinline__ uint32_t __vimax_s32_relu(uint32_t a,uint32_t b) { return packed<32,true,Op::Max,true>(a,b); }
__device__ __forceinline__ uint32_t __vimin3_s16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::Min3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin3_s16x2_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<16,true,Op::Min3,true>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin3_s32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::Min3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin3_s32_relu(uint32_t a,uint32_t b,uint32_t c) { return packed<32,true,Op::Min3,true>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin3_u16x2(uint32_t a,uint32_t b,uint32_t c) { return packed<16,false,Op::Min3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin3_u32(uint32_t a,uint32_t b,uint32_t c) { return packed<32,false,Op::Min3,false>(a,b,c); }
__device__ __forceinline__ uint32_t __vimin_s16x2_relu(uint32_t a,uint32_t b) { return packed<16,true,Op::Min,true>(a,b); }
__device__ __forceinline__ uint32_t __vimin_s32_relu(uint32_t a,uint32_t b) { return packed<32,true,Op::Min,true>(a,b); }
__device__ __forceinline__ uint32_t __vmaxs2(uint32_t a,uint32_t b) { return packed<16,true,Op::Max,false>(a,b); }
__device__ __forceinline__ uint32_t __vmaxs4(uint32_t a,uint32_t b) { return packed<8,true,Op::Max,false>(a,b); }
__device__ __forceinline__ uint32_t __vmaxu2(uint32_t a,uint32_t b) { return packed<16,false,Op::Max,false>(a,b); }
__device__ __forceinline__ uint32_t __vmaxu4(uint32_t a,uint32_t b) { return packed<8,false,Op::Max,false>(a,b); }
__device__ __forceinline__ uint32_t __vmins2(uint32_t a,uint32_t b) { return packed<16,true,Op::Min,false>(a,b); }
__device__ __forceinline__ uint32_t __vmins4(uint32_t a,uint32_t b) { return packed<8,true,Op::Min,false>(a,b); }
__device__ __forceinline__ uint32_t __vminu2(uint32_t a,uint32_t b) { return packed<16,false,Op::Min,false>(a,b); }
__device__ __forceinline__ uint32_t __vminu4(uint32_t a,uint32_t b) { return packed<8,false,Op::Min,false>(a,b); }
__device__ __forceinline__ uint32_t __vneg2(uint32_t a) { return packed<16,true,Op::Neg,false>(a); }
__device__ __forceinline__ uint32_t __vneg4(uint32_t a) { return packed<8,true,Op::Neg,false>(a); }
__device__ __forceinline__ uint32_t __vnegss2(uint32_t a) { return packed<16,true,Op::NegSat,false>(a); }
__device__ __forceinline__ uint32_t __vnegss4(uint32_t a) { return packed<8,true,Op::NegSat,false>(a); }
__device__ __forceinline__ uint32_t __vsads2(uint32_t a,uint32_t b) { return packed<16,true,Op::Sad,false>(a,b); }
__device__ __forceinline__ uint32_t __vsads4(uint32_t a,uint32_t b) { return packed<8,true,Op::Sad,false>(a,b); }
__device__ __forceinline__ uint32_t __vsadu2(uint32_t a,uint32_t b) { return packed<16,false,Op::Sad,false>(a,b); }
__device__ __forceinline__ uint32_t __vsadu4(uint32_t a,uint32_t b) { return packed<8,false,Op::Sad,false>(a,b); }
__device__ __forceinline__ uint32_t __vseteq2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetEq,false>(a,b); }
__device__ __forceinline__ uint32_t __vseteq4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetEq,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetges2(uint32_t a,uint32_t b) { return packed<16,true,Op::SetGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetges4(uint32_t a,uint32_t b) { return packed<8,true,Op::SetGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgeu2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgeu4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetGe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgts2(uint32_t a,uint32_t b) { return packed<16,true,Op::SetGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgts4(uint32_t a,uint32_t b) { return packed<8,true,Op::SetGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgtu2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetgtu4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetGt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetles2(uint32_t a,uint32_t b) { return packed<16,true,Op::SetLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetles4(uint32_t a,uint32_t b) { return packed<8,true,Op::SetLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetleu2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetleu4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetLe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetlts2(uint32_t a,uint32_t b) { return packed<16,true,Op::SetLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetlts4(uint32_t a,uint32_t b) { return packed<8,true,Op::SetLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetltu2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetltu4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetLt,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetne2(uint32_t a,uint32_t b) { return packed<16,false,Op::SetNe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsetne4(uint32_t a,uint32_t b) { return packed<8,false,Op::SetNe,false>(a,b); }
__device__ __forceinline__ uint32_t __vsub2(uint32_t a,uint32_t b) { return packed<16,false,Op::Sub,false>(a,b); }
__device__ __forceinline__ uint32_t __vsub4(uint32_t a,uint32_t b) { return packed<8,false,Op::Sub,false>(a,b); }
__device__ __forceinline__ uint32_t __vsubss2(uint32_t a,uint32_t b) { return packed<16,true,Op::SubSatS,false>(a,b); }
__device__ __forceinline__ uint32_t __vsubss4(uint32_t a,uint32_t b) { return packed<8,true,Op::SubSatS,false>(a,b); }
__device__ __forceinline__ uint32_t __vsubus2(uint32_t a,uint32_t b) { return packed<16,false,Op::SubSatU,false>(a,b); }
__device__ __forceinline__ uint32_t __vsubus4(uint32_t a,uint32_t b) { return packed<8,false,Op::SubSatU,false>(a,b); }
}
