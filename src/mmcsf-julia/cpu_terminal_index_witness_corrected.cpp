#include "configured/util.h"
int main(){Tensor x;x.ndims=3;x.totNnz=2;x.dims=new size_t[3]{3,3,3};x.modeOrder={0,1,2};x.inds=new size_t*[3];x.inds[0]=new size_t[2]{0,0};x.inds[1]=new size_t[2]{0,0};x.inds[2]=new size_t[2]{0,1};x.vals=new float[2]{1,2};Options o;create_HCSR(x,o);return 0;}
