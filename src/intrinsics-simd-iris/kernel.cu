#include <cuda_runtime.h>
#include <cstdint>
#include "operations.hpp"
extern "C" __global__ void simd_intrinsics(const int n,
                     const unsigned int* input,
                           unsigned int* output)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

  unsigned int a = input[i];
  unsigned int b = a + ((i % 2) ? 1 : -1);
  unsigned int c = a ^ b; 
  unsigned int r;

  r  = __vabs2(a);
  r ^= __vabs4(a);
  
  r ^= __vabsdiffs2(a, b);
  r ^= __vabsdiffs4(a, b);
  r ^= __vabsdiffu2(a, b);
  r ^= __vabsdiffu4(a, b);
  
  r ^= __vabsss2(a);
  r ^= __vabsss4(a);
  
  r ^= __vadd2(a, b);
  r ^= __vadd4(a, b);
  
  r ^= __vaddss2(a, b);
  r ^= __vaddss4(a, b);
  r ^= __vaddus2(a, b);
  r ^= __vaddus4(a, b);
  
  r ^= __vavgs2(a, b);
  r ^= __vavgs4(a, b);
  r ^= __vavgu2(a, b);
  r ^= __vavgu4(a, b);
  
  r ^= __vcmpeq2(a, b);
  r ^= __vcmpeq4(a, b);
  
  r ^= __vcmpges2(a, b);
  r ^= __vcmpges4(a, b);
  r ^= __vcmpgeu2(a, b);
  r ^= __vcmpgeu4(a, b);
  
  r ^= __vcmpgts2(a, b);
  r ^= __vcmpgts4(a, b);
  r ^= __vcmpgtu2(a, b);
  r ^= __vcmpgtu4(a, b);
  
  r ^= __vcmples2(a, b);
  r ^= __vcmples4(a, b);
  r ^= __vcmpleu2(a, b);
  r ^= __vcmpleu4(a, b);
  
  r ^= __vcmplts2(a, b);
  r ^= __vcmplts4(a, b);
  r ^= __vcmpltu2(a, b);
  r ^= __vcmpltu4(a, b);
  
  r ^= __vcmpne2(a, b);
  r ^= __vcmpne4(a, b);
  
  r ^= __vhaddu2(a, b);
  r ^= __vhaddu4(a, b);
  
  r ^= __viaddmax_s16x2(a, b, c);
  r ^= __viaddmax_s16x2_relu(a, b, c);
  r ^= __viaddmax_s32(a, b, c);
  r ^= __viaddmax_s32_relu(a, b, c);
  r ^= __viaddmax_u16x2(a, b, c);
  r ^= __viaddmax_u32(a, b, c);
  
  r ^= __viaddmin_s16x2(a, b, c);
  r ^= __viaddmin_s16x2_relu(a, b, c);
  r ^= __viaddmin_s32(a, b, c);
  r ^= __viaddmin_s32_relu(a, b, c);
  r ^= __viaddmin_u16x2(a, b, c);
  r ^= __viaddmin_u32(a, b, c);
  
  bool pred   ;
  bool pred_hi;
  bool pred_lo;

  r ^= __vibmax_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmax_s32(a, b, &pred);
  r ^= __vibmax_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmax_u32(a, b, &pred);
  
  r ^= __vibmin_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmin_s32(a, b, &pred);
  r ^= __vibmin_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= __vibmin_u32(a, b, &pred);
  
  r ^= __vimax3_s16x2(a, b, c);
  r ^= __vimax3_s16x2_relu(a, b, c);
  r ^= __vimax3_s32(a, b, c);
  r ^= __vimax3_s32_relu(a, b, c);
  r ^= __vimax3_u16x2(a, b, c);
  r ^= __vimax3_u32(a, b, c);
  
  r ^= __vimax_s16x2_relu(a, b);
  r ^= __vimax_s32_relu(a, b);
  
  r ^= __vimin3_s16x2(a, b, c);
  r ^= __vimin3_s16x2_relu(a, b, c);
  r ^= __vimin3_s32(a, b, c);
  r ^= __vimin3_s32_relu(a, b, c);
  r ^= __vimin3_u16x2(a, b, c);
  r ^= __vimin3_u32(a, b, c);
  
  r ^= __vimin_s16x2_relu(a, b);
  r ^= __vimin_s32_relu(a, b);
  
  r ^= __vmaxs2(a, b);
  r ^= __vmaxs4(a, b);
  r ^= __vmaxu2(a, b);
  r ^= __vmaxu4(a, b);
  
  r ^= __vmins2(a, b);
  r ^= __vmins4(a, b);
  r ^= __vminu2(a, b);
  r ^= __vminu4(a, b);
  
  r ^= __vneg2(a);
  r ^= __vneg4(a);
  r ^= __vnegss2(a);
  r ^= __vnegss4(a);
  
  r ^= __vsads2(a, b);
  r ^= __vsads4(a, b);
  r ^= __vsadu2(a, b);
  r ^= __vsadu4(a, b);
  
  r ^= __vseteq2(a, b);
  r ^= __vseteq4(a, b);
  
  r ^= __vsetges2(a, b);
  r ^= __vsetges4(a, b);
  r ^= __vsetgeu2(a, b);
  r ^= __vsetgeu4(a, b);
  
  r ^= __vsetgts2(a, b);
  r ^= __vsetgts4(a, b);
  r ^= __vsetgtu2(a, b);
  r ^= __vsetgtu4(a, b);
  
  r ^= __vsetles2(a, b);
  r ^= __vsetles4(a, b);
  r ^= __vsetleu2(a, b);
  r ^= __vsetleu4(a, b);
  
  r ^= __vsetlts2(a, b);
  r ^= __vsetlts4(a, b);
  r ^= __vsetltu2(a, b);
  r ^= __vsetltu4(a, b);
  
  r ^= __vsetne2(a, b);
  r ^= __vsetne4(a, b);
  
  r ^= __vsub2(a, b);
  r ^= __vsub4(a, b);
  r ^= __vsubss2(a, b);
  r ^= __vsubss4(a, b);
  r ^= __vsubus2(a, b);
  r ^= __vsubus4(a, b);

  output[i] = r;
}

extern "C" __global__ void individual_checks(const int n,const uint32_t* triples,uint32_t* output) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  uint32_t a=triples[i*3],b=triples[i*3+1],c=triples[i*3+2];
  bool pred=false,pred_hi=false,pred_lo=false;
  output[(size_t(i)*OP_COUNT+0)*2]=__vabs2(a);
  output[(size_t(i)*OP_COUNT+0)*2+1]=0;
  output[(size_t(i)*OP_COUNT+1)*2]=__vabs4(a);
  output[(size_t(i)*OP_COUNT+1)*2+1]=0;
  output[(size_t(i)*OP_COUNT+2)*2]=__vabsdiffs2(a, b);
  output[(size_t(i)*OP_COUNT+2)*2+1]=0;
  output[(size_t(i)*OP_COUNT+3)*2]=__vabsdiffs4(a, b);
  output[(size_t(i)*OP_COUNT+3)*2+1]=0;
  output[(size_t(i)*OP_COUNT+4)*2]=__vabsdiffu2(a, b);
  output[(size_t(i)*OP_COUNT+4)*2+1]=0;
  output[(size_t(i)*OP_COUNT+5)*2]=__vabsdiffu4(a, b);
  output[(size_t(i)*OP_COUNT+5)*2+1]=0;
  output[(size_t(i)*OP_COUNT+6)*2]=__vabsss2(a);
  output[(size_t(i)*OP_COUNT+6)*2+1]=0;
  output[(size_t(i)*OP_COUNT+7)*2]=__vabsss4(a);
  output[(size_t(i)*OP_COUNT+7)*2+1]=0;
  output[(size_t(i)*OP_COUNT+8)*2]=__vadd2(a, b);
  output[(size_t(i)*OP_COUNT+8)*2+1]=0;
  output[(size_t(i)*OP_COUNT+9)*2]=__vadd4(a, b);
  output[(size_t(i)*OP_COUNT+9)*2+1]=0;
  output[(size_t(i)*OP_COUNT+10)*2]=__vaddss2(a, b);
  output[(size_t(i)*OP_COUNT+10)*2+1]=0;
  output[(size_t(i)*OP_COUNT+11)*2]=__vaddss4(a, b);
  output[(size_t(i)*OP_COUNT+11)*2+1]=0;
  output[(size_t(i)*OP_COUNT+12)*2]=__vaddus2(a, b);
  output[(size_t(i)*OP_COUNT+12)*2+1]=0;
  output[(size_t(i)*OP_COUNT+13)*2]=__vaddus4(a, b);
  output[(size_t(i)*OP_COUNT+13)*2+1]=0;
  output[(size_t(i)*OP_COUNT+14)*2]=__vavgs2(a, b);
  output[(size_t(i)*OP_COUNT+14)*2+1]=0;
  output[(size_t(i)*OP_COUNT+15)*2]=__vavgs4(a, b);
  output[(size_t(i)*OP_COUNT+15)*2+1]=0;
  output[(size_t(i)*OP_COUNT+16)*2]=__vavgu2(a, b);
  output[(size_t(i)*OP_COUNT+16)*2+1]=0;
  output[(size_t(i)*OP_COUNT+17)*2]=__vavgu4(a, b);
  output[(size_t(i)*OP_COUNT+17)*2+1]=0;
  output[(size_t(i)*OP_COUNT+18)*2]=__vcmpeq2(a, b);
  output[(size_t(i)*OP_COUNT+18)*2+1]=0;
  output[(size_t(i)*OP_COUNT+19)*2]=__vcmpeq4(a, b);
  output[(size_t(i)*OP_COUNT+19)*2+1]=0;
  output[(size_t(i)*OP_COUNT+20)*2]=__vcmpges2(a, b);
  output[(size_t(i)*OP_COUNT+20)*2+1]=0;
  output[(size_t(i)*OP_COUNT+21)*2]=__vcmpges4(a, b);
  output[(size_t(i)*OP_COUNT+21)*2+1]=0;
  output[(size_t(i)*OP_COUNT+22)*2]=__vcmpgeu2(a, b);
  output[(size_t(i)*OP_COUNT+22)*2+1]=0;
  output[(size_t(i)*OP_COUNT+23)*2]=__vcmpgeu4(a, b);
  output[(size_t(i)*OP_COUNT+23)*2+1]=0;
  output[(size_t(i)*OP_COUNT+24)*2]=__vcmpgts2(a, b);
  output[(size_t(i)*OP_COUNT+24)*2+1]=0;
  output[(size_t(i)*OP_COUNT+25)*2]=__vcmpgts4(a, b);
  output[(size_t(i)*OP_COUNT+25)*2+1]=0;
  output[(size_t(i)*OP_COUNT+26)*2]=__vcmpgtu2(a, b);
  output[(size_t(i)*OP_COUNT+26)*2+1]=0;
  output[(size_t(i)*OP_COUNT+27)*2]=__vcmpgtu4(a, b);
  output[(size_t(i)*OP_COUNT+27)*2+1]=0;
  output[(size_t(i)*OP_COUNT+28)*2]=__vcmples2(a, b);
  output[(size_t(i)*OP_COUNT+28)*2+1]=0;
  output[(size_t(i)*OP_COUNT+29)*2]=__vcmples4(a, b);
  output[(size_t(i)*OP_COUNT+29)*2+1]=0;
  output[(size_t(i)*OP_COUNT+30)*2]=__vcmpleu2(a, b);
  output[(size_t(i)*OP_COUNT+30)*2+1]=0;
  output[(size_t(i)*OP_COUNT+31)*2]=__vcmpleu4(a, b);
  output[(size_t(i)*OP_COUNT+31)*2+1]=0;
  output[(size_t(i)*OP_COUNT+32)*2]=__vcmplts2(a, b);
  output[(size_t(i)*OP_COUNT+32)*2+1]=0;
  output[(size_t(i)*OP_COUNT+33)*2]=__vcmplts4(a, b);
  output[(size_t(i)*OP_COUNT+33)*2+1]=0;
  output[(size_t(i)*OP_COUNT+34)*2]=__vcmpltu2(a, b);
  output[(size_t(i)*OP_COUNT+34)*2+1]=0;
  output[(size_t(i)*OP_COUNT+35)*2]=__vcmpltu4(a, b);
  output[(size_t(i)*OP_COUNT+35)*2+1]=0;
  output[(size_t(i)*OP_COUNT+36)*2]=__vcmpne2(a, b);
  output[(size_t(i)*OP_COUNT+36)*2+1]=0;
  output[(size_t(i)*OP_COUNT+37)*2]=__vcmpne4(a, b);
  output[(size_t(i)*OP_COUNT+37)*2+1]=0;
  output[(size_t(i)*OP_COUNT+38)*2]=__vhaddu2(a, b);
  output[(size_t(i)*OP_COUNT+38)*2+1]=0;
  output[(size_t(i)*OP_COUNT+39)*2]=__vhaddu4(a, b);
  output[(size_t(i)*OP_COUNT+39)*2+1]=0;
  output[(size_t(i)*OP_COUNT+40)*2]=__viaddmax_s16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+40)*2+1]=0;
  output[(size_t(i)*OP_COUNT+41)*2]=__viaddmax_s16x2_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+41)*2+1]=0;
  output[(size_t(i)*OP_COUNT+42)*2]=__viaddmax_s32(a, b, c);
  output[(size_t(i)*OP_COUNT+42)*2+1]=0;
  output[(size_t(i)*OP_COUNT+43)*2]=__viaddmax_s32_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+43)*2+1]=0;
  output[(size_t(i)*OP_COUNT+44)*2]=__viaddmax_u16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+44)*2+1]=0;
  output[(size_t(i)*OP_COUNT+45)*2]=__viaddmax_u32(a, b, c);
  output[(size_t(i)*OP_COUNT+45)*2+1]=0;
  output[(size_t(i)*OP_COUNT+46)*2]=__viaddmin_s16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+46)*2+1]=0;
  output[(size_t(i)*OP_COUNT+47)*2]=__viaddmin_s16x2_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+47)*2+1]=0;
  output[(size_t(i)*OP_COUNT+48)*2]=__viaddmin_s32(a, b, c);
  output[(size_t(i)*OP_COUNT+48)*2+1]=0;
  output[(size_t(i)*OP_COUNT+49)*2]=__viaddmin_s32_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+49)*2+1]=0;
  output[(size_t(i)*OP_COUNT+50)*2]=__viaddmin_u16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+50)*2+1]=0;
  output[(size_t(i)*OP_COUNT+51)*2]=__viaddmin_u32(a, b, c);
  output[(size_t(i)*OP_COUNT+51)*2+1]=0;
  output[(size_t(i)*OP_COUNT+52)*2]=__vibmax_s16x2(a, b, &pred_hi, &pred_lo);
  output[(size_t(i)*OP_COUNT+52)*2+1]=uint32_t(pred_lo)|(uint32_t(pred_hi)<<1);
  output[(size_t(i)*OP_COUNT+53)*2]=__vibmax_s32(a, b, &pred);
  output[(size_t(i)*OP_COUNT+53)*2+1]=uint32_t(pred);
  output[(size_t(i)*OP_COUNT+54)*2]=__vibmax_u16x2(a, b, &pred_hi, &pred_lo);
  output[(size_t(i)*OP_COUNT+54)*2+1]=uint32_t(pred_lo)|(uint32_t(pred_hi)<<1);
  output[(size_t(i)*OP_COUNT+55)*2]=__vibmax_u32(a, b, &pred);
  output[(size_t(i)*OP_COUNT+55)*2+1]=uint32_t(pred);
  output[(size_t(i)*OP_COUNT+56)*2]=__vibmin_s16x2(a, b, &pred_hi, &pred_lo);
  output[(size_t(i)*OP_COUNT+56)*2+1]=uint32_t(pred_lo)|(uint32_t(pred_hi)<<1);
  output[(size_t(i)*OP_COUNT+57)*2]=__vibmin_s32(a, b, &pred);
  output[(size_t(i)*OP_COUNT+57)*2+1]=uint32_t(pred);
  output[(size_t(i)*OP_COUNT+58)*2]=__vibmin_u16x2(a, b, &pred_hi, &pred_lo);
  output[(size_t(i)*OP_COUNT+58)*2+1]=uint32_t(pred_lo)|(uint32_t(pred_hi)<<1);
  output[(size_t(i)*OP_COUNT+59)*2]=__vibmin_u32(a, b, &pred);
  output[(size_t(i)*OP_COUNT+59)*2+1]=uint32_t(pred);
  output[(size_t(i)*OP_COUNT+60)*2]=__vimax3_s16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+60)*2+1]=0;
  output[(size_t(i)*OP_COUNT+61)*2]=__vimax3_s16x2_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+61)*2+1]=0;
  output[(size_t(i)*OP_COUNT+62)*2]=__vimax3_s32(a, b, c);
  output[(size_t(i)*OP_COUNT+62)*2+1]=0;
  output[(size_t(i)*OP_COUNT+63)*2]=__vimax3_s32_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+63)*2+1]=0;
  output[(size_t(i)*OP_COUNT+64)*2]=__vimax3_u16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+64)*2+1]=0;
  output[(size_t(i)*OP_COUNT+65)*2]=__vimax3_u32(a, b, c);
  output[(size_t(i)*OP_COUNT+65)*2+1]=0;
  output[(size_t(i)*OP_COUNT+66)*2]=__vimax_s16x2_relu(a, b);
  output[(size_t(i)*OP_COUNT+66)*2+1]=0;
  output[(size_t(i)*OP_COUNT+67)*2]=__vimax_s32_relu(a, b);
  output[(size_t(i)*OP_COUNT+67)*2+1]=0;
  output[(size_t(i)*OP_COUNT+68)*2]=__vimin3_s16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+68)*2+1]=0;
  output[(size_t(i)*OP_COUNT+69)*2]=__vimin3_s16x2_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+69)*2+1]=0;
  output[(size_t(i)*OP_COUNT+70)*2]=__vimin3_s32(a, b, c);
  output[(size_t(i)*OP_COUNT+70)*2+1]=0;
  output[(size_t(i)*OP_COUNT+71)*2]=__vimin3_s32_relu(a, b, c);
  output[(size_t(i)*OP_COUNT+71)*2+1]=0;
  output[(size_t(i)*OP_COUNT+72)*2]=__vimin3_u16x2(a, b, c);
  output[(size_t(i)*OP_COUNT+72)*2+1]=0;
  output[(size_t(i)*OP_COUNT+73)*2]=__vimin3_u32(a, b, c);
  output[(size_t(i)*OP_COUNT+73)*2+1]=0;
  output[(size_t(i)*OP_COUNT+74)*2]=__vimin_s16x2_relu(a, b);
  output[(size_t(i)*OP_COUNT+74)*2+1]=0;
  output[(size_t(i)*OP_COUNT+75)*2]=__vimin_s32_relu(a, b);
  output[(size_t(i)*OP_COUNT+75)*2+1]=0;
  output[(size_t(i)*OP_COUNT+76)*2]=__vmaxs2(a, b);
  output[(size_t(i)*OP_COUNT+76)*2+1]=0;
  output[(size_t(i)*OP_COUNT+77)*2]=__vmaxs4(a, b);
  output[(size_t(i)*OP_COUNT+77)*2+1]=0;
  output[(size_t(i)*OP_COUNT+78)*2]=__vmaxu2(a, b);
  output[(size_t(i)*OP_COUNT+78)*2+1]=0;
  output[(size_t(i)*OP_COUNT+79)*2]=__vmaxu4(a, b);
  output[(size_t(i)*OP_COUNT+79)*2+1]=0;
  output[(size_t(i)*OP_COUNT+80)*2]=__vmins2(a, b);
  output[(size_t(i)*OP_COUNT+80)*2+1]=0;
  output[(size_t(i)*OP_COUNT+81)*2]=__vmins4(a, b);
  output[(size_t(i)*OP_COUNT+81)*2+1]=0;
  output[(size_t(i)*OP_COUNT+82)*2]=__vminu2(a, b);
  output[(size_t(i)*OP_COUNT+82)*2+1]=0;
  output[(size_t(i)*OP_COUNT+83)*2]=__vminu4(a, b);
  output[(size_t(i)*OP_COUNT+83)*2+1]=0;
  output[(size_t(i)*OP_COUNT+84)*2]=__vneg2(a);
  output[(size_t(i)*OP_COUNT+84)*2+1]=0;
  output[(size_t(i)*OP_COUNT+85)*2]=__vneg4(a);
  output[(size_t(i)*OP_COUNT+85)*2+1]=0;
  output[(size_t(i)*OP_COUNT+86)*2]=__vnegss2(a);
  output[(size_t(i)*OP_COUNT+86)*2+1]=0;
  output[(size_t(i)*OP_COUNT+87)*2]=__vnegss4(a);
  output[(size_t(i)*OP_COUNT+87)*2+1]=0;
  output[(size_t(i)*OP_COUNT+88)*2]=__vsads2(a, b);
  output[(size_t(i)*OP_COUNT+88)*2+1]=0;
  output[(size_t(i)*OP_COUNT+89)*2]=__vsads4(a, b);
  output[(size_t(i)*OP_COUNT+89)*2+1]=0;
  output[(size_t(i)*OP_COUNT+90)*2]=__vsadu2(a, b);
  output[(size_t(i)*OP_COUNT+90)*2+1]=0;
  output[(size_t(i)*OP_COUNT+91)*2]=__vsadu4(a, b);
  output[(size_t(i)*OP_COUNT+91)*2+1]=0;
  output[(size_t(i)*OP_COUNT+92)*2]=__vseteq2(a, b);
  output[(size_t(i)*OP_COUNT+92)*2+1]=0;
  output[(size_t(i)*OP_COUNT+93)*2]=__vseteq4(a, b);
  output[(size_t(i)*OP_COUNT+93)*2+1]=0;
  output[(size_t(i)*OP_COUNT+94)*2]=__vsetges2(a, b);
  output[(size_t(i)*OP_COUNT+94)*2+1]=0;
  output[(size_t(i)*OP_COUNT+95)*2]=__vsetges4(a, b);
  output[(size_t(i)*OP_COUNT+95)*2+1]=0;
  output[(size_t(i)*OP_COUNT+96)*2]=__vsetgeu2(a, b);
  output[(size_t(i)*OP_COUNT+96)*2+1]=0;
  output[(size_t(i)*OP_COUNT+97)*2]=__vsetgeu4(a, b);
  output[(size_t(i)*OP_COUNT+97)*2+1]=0;
  output[(size_t(i)*OP_COUNT+98)*2]=__vsetgts2(a, b);
  output[(size_t(i)*OP_COUNT+98)*2+1]=0;
  output[(size_t(i)*OP_COUNT+99)*2]=__vsetgts4(a, b);
  output[(size_t(i)*OP_COUNT+99)*2+1]=0;
  output[(size_t(i)*OP_COUNT+100)*2]=__vsetgtu2(a, b);
  output[(size_t(i)*OP_COUNT+100)*2+1]=0;
  output[(size_t(i)*OP_COUNT+101)*2]=__vsetgtu4(a, b);
  output[(size_t(i)*OP_COUNT+101)*2+1]=0;
  output[(size_t(i)*OP_COUNT+102)*2]=__vsetles2(a, b);
  output[(size_t(i)*OP_COUNT+102)*2+1]=0;
  output[(size_t(i)*OP_COUNT+103)*2]=__vsetles4(a, b);
  output[(size_t(i)*OP_COUNT+103)*2+1]=0;
  output[(size_t(i)*OP_COUNT+104)*2]=__vsetleu2(a, b);
  output[(size_t(i)*OP_COUNT+104)*2+1]=0;
  output[(size_t(i)*OP_COUNT+105)*2]=__vsetleu4(a, b);
  output[(size_t(i)*OP_COUNT+105)*2+1]=0;
  output[(size_t(i)*OP_COUNT+106)*2]=__vsetlts2(a, b);
  output[(size_t(i)*OP_COUNT+106)*2+1]=0;
  output[(size_t(i)*OP_COUNT+107)*2]=__vsetlts4(a, b);
  output[(size_t(i)*OP_COUNT+107)*2+1]=0;
  output[(size_t(i)*OP_COUNT+108)*2]=__vsetltu2(a, b);
  output[(size_t(i)*OP_COUNT+108)*2+1]=0;
  output[(size_t(i)*OP_COUNT+109)*2]=__vsetltu4(a, b);
  output[(size_t(i)*OP_COUNT+109)*2+1]=0;
  output[(size_t(i)*OP_COUNT+110)*2]=__vsetne2(a, b);
  output[(size_t(i)*OP_COUNT+110)*2+1]=0;
  output[(size_t(i)*OP_COUNT+111)*2]=__vsetne4(a, b);
  output[(size_t(i)*OP_COUNT+111)*2+1]=0;
  output[(size_t(i)*OP_COUNT+112)*2]=__vsub2(a, b);
  output[(size_t(i)*OP_COUNT+112)*2+1]=0;
  output[(size_t(i)*OP_COUNT+113)*2]=__vsub4(a, b);
  output[(size_t(i)*OP_COUNT+113)*2+1]=0;
  output[(size_t(i)*OP_COUNT+114)*2]=__vsubss2(a, b);
  output[(size_t(i)*OP_COUNT+114)*2+1]=0;
  output[(size_t(i)*OP_COUNT+115)*2]=__vsubss4(a, b);
  output[(size_t(i)*OP_COUNT+115)*2+1]=0;
  output[(size_t(i)*OP_COUNT+116)*2]=__vsubus2(a, b);
  output[(size_t(i)*OP_COUNT+116)*2+1]=0;
  output[(size_t(i)*OP_COUNT+117)*2]=__vsubus4(a, b);
  output[(size_t(i)*OP_COUNT+117)*2+1]=0;
}
