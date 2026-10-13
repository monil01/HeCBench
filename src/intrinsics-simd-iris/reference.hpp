#pragma once
#include "operations.hpp"
#include <algorithm>
#include <cstdlib>
// Independent host model: mathematical modulo/extraction, widened arithmetic.
static int64_t ref_signed(int64_t v,int bits) {
  int64_t base=int64_t(1)<<bits;
  v%=base;if(v<0)v+=base;
  return v>=base/2?v-base:v;
}
static uint32_t reference(int id,uint32_t a,uint32_t b=0,uint32_t c=0,uint32_t* pred=nullptr) {
  auto s=specs[id];int64_t base=int64_t(1)<<s.bits,mask=base-1,result=0,sad=0;
  if(pred)*pred=0;
  for(int lane=0;lane<32/s.bits;++lane) {
    int64_t place=int64_t(1)<<(lane*s.bits);
    int64_t x=(a/place)%base,y=(b/place)%base,z=(c/place)%base;
    if(s.sign){x=ref_signed(x,s.bits);y=ref_signed(y,s.bits);z=ref_signed(z,s.bits);}
    int64_t v=0;
    switch(s.op) {
      case Op::Abs: v=std::abs(x);break;
      case Op::AbsDiff: v=std::abs(x-y);break;
      case Op::AbsSat: v=std::min(std::abs(x),base/2-1);break;
      case Op::Add: v=x+y;break;
      case Op::AddSatS: v=std::clamp(x+y,-base/2,base/2-1);break;
      case Op::AddSatU: v=std::clamp(x+y,int64_t(0),mask);break;
      case Op::AvgS: v=x+y<0?-((-(x+y)+1)/2):(x+y+1)/2;break;
      case Op::AvgU: v=(x+y+1)/2;break;
      case Op::MaskEq: v=(x==y)?mask:0;break;
      case Op::MaskGe: v=(x>=y)?mask:0;break;
      case Op::MaskGt: v=(x>y)?mask:0;break;
      case Op::MaskLe: v=(x<=y)?mask:0;break;
      case Op::MaskLt: v=(x<y)?mask:0;break;
      case Op::MaskNe: v=(x!=y)?mask:0;break;
      case Op::Half: v=(x+y)/2;break;
      case Op::AddMax: v=std::max(s.sign?ref_signed(x+y,s.bits):(x+y)%base,z);break;
      case Op::AddMin: v=std::min(s.sign?ref_signed(x+y,s.bits):(x+y)%base,z);break;
      case Op::PredMax: v=std::max(x,y);break;
      case Op::PredMin: v=std::min(x,y);break;
      case Op::Max3: v=std::max({x,y,z});break;
      case Op::Max: v=std::max(x,y);break;
      case Op::Min3: v=std::min({x,y,z});break;
      case Op::Min: v=std::min(x,y);break;
      case Op::Neg: v=-x;break;
      case Op::NegSat: v=std::min(-x,base/2-1);break;
      case Op::Sad: v=std::abs(x-y);break;
      case Op::SetEq: v=(x==y)?1:0;break;
      case Op::SetGe: v=(x>=y)?1:0;break;
      case Op::SetGt: v=(x>y)?1:0;break;
      case Op::SetLe: v=(x<=y)?1:0;break;
      case Op::SetLt: v=(x<y)?1:0;break;
      case Op::SetNe: v=(x!=y)?1:0;break;
      case Op::Sub: v=x-y;break;
      case Op::SubSatS: v=std::clamp(x-y,-base/2,base/2-1);break;
      case Op::SubSatU: v=std::clamp(x-y,int64_t(0),mask);break;
    }
    if(s.relu)v=std::max(v,int64_t(0));
    if(pred&&(s.op==Op::PredMax||s.op==Op::PredMin))
      *pred|=uint32_t(s.op==Op::PredMax?x>=y:x<=y)<<lane;
    if(s.op==Op::Sad)sad+=v;
    else { v%=base;if(v<0)v+=base;result+=v*place; }
  }
  return uint32_t(s.op==Op::Sad?sad:result);
}
static uint32_t reference_full(uint32_t a,int i) {
  uint32_t b=a+((i%2)?1u:~0u),c=a^b,r=0;
  for(int id=0;id<OP_COUNT;++id)r^=reference(id,a,b,c);
  return r;
}
