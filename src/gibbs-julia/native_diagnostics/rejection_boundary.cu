#define main original_main
#include "../../gibbs-cuda/main.cu"
#undef main
#include <fstream>
__global__ void diagnostic(curandState*state,float shape,float rate,float*out){curandState s=*state;float d=shape-1.f/3.f,c=1.f/sqrtf(9.f*d);float z=curand_normal(&s);float v=powf(1.f+c*z,3.f);float u=curand_uniform(&s);float rhs=.5f*z*z+d-d*v+d*logf(v);out[0]=z;out[1]=u;out[2]=d;out[3]=c;out[4]=v;out[5]=logf(u);out[6]=rhs;out[7]=logf(u)-rhs;out[8]=rgamma(state,shape,rate);}
int main(int argc,char**argv){curandState hs;float shape,rate;std::ifstream f(argv[1],std::ios::binary);f.read((char*)&hs,48);f.read((char*)&shape,4);f.read((char*)&rate,4);curandState*s;float*out;cudaMalloc(&s,48);cudaMalloc(&out,9*4);cudaMemcpy(s,&hs,48,cudaMemcpyHostToDevice);diagnostic<<<1,1>>>(s,shape,rate,out);int status=cudaDeviceSynchronize();float h[9];cudaMemcpy(h,out,36,cudaMemcpyDeviceToHost);for(int i=0;i<9;i++)printf("boundary_field%d=%.17g\n",i,h[i]);printf("status%d\n",status);cudaFree(s);cudaFree(out);return status;}
