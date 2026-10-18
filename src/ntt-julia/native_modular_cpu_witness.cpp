#include "../ntt-cuda/modP.h"
extern "C" unsigned long shift_original(unsigned long x,int l){return _ls_modP(x,l);}
extern "C" unsigned long multiply_original(unsigned long x,unsigned long y){return _mul_modP(x,y,valP);}
extern "C" void ntt8_original(unsigned long*x){ntt8(x);}
