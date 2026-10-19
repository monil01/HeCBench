// Independent conditional-gamma evaluator using scalar CPU math and explicit XORWOW.
// No vendor normal/gamma call; retains algorithm's cached pair and finite arithmetic.
#include <cstdint>
#include <cmath>
#include <cstring>
struct StudyRNG{uint32_t d,v[5];int flag,flag_double;float extra;double extra_double;};
static_assert(sizeof(StudyRNG)==48,"pinnedcurandABI");
static uint32_t word(StudyRNG&s){uint32_t t=s.v[0]^(s.v[0]>>2);for(int i=0;i<4;i++)s.v[i]=s.v[i+1];s.v[4]=(s.v[4]^(s.v[4]<<4))^(t^(t<<1));s.d+=362437;return s.v[4]+s.d;}
static float uniform(StudyRNG&s){return fmaf(float(word(s)),2.3283064e-10f,2.3283064e-10f/2.f);}
static float normal(StudyRNG&s){if(s.flag==1){s.flag=0;return s.extra;}uint32_t x=word(s),y=word(s);float u=fmaf(float(x),2.3283064e-10f,2.3283064e-10f/2.f);float c=2.3283064e-10f*6.2831855f;float angle=fmaf(float(y),c,c/2.f);float radius=sqrtf(-2.f*logf(u));s.extra=cosf(angle)*radius;s.flag=1;return sinf(angle)*radius;}
static float gamma_cpu(StudyRNG&s,float shape,float rate){float d=shape-1.f/3.f,c=1.f/sqrtf(9.f*d);for(int attempt=0;attempt<10000;attempt++){float z=normal(s);if(z> -1.f/c){float v=powf(fmaf(c,z,1.f),3.f);float u=uniform(s);float lhs=logf(u);float rhs=(.5f*z*z+d-d*v)+d*logf(v);if(!(lhs>rhs))return d*v/rate;}}return NAN;}
static float conditional_cpu(StudyRNG&s,float shape,float rate){if(shape<1.f){float x=gamma_cpu(s,shape+1.f,rate);return x*powf(uniform(s),1.f/shape);}return gamma_cpu(s,shape,rate);}
