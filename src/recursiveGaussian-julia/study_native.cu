#include <cuda_runtime.h>
#include <cmath>
#include "../recursiveGaussian-cuda/helper_math.h"
#include "../recursiveGaussian-cuda/main.h"
// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
__device__
float4 rgbaUintToFloat4(const unsigned int uiPackedRGBA)
{
    float4 rgba;
    rgba.x = uiPackedRGBA & 0xff;
    rgba.y = (uiPackedRGBA >> 8) & 0xff;
    rgba.z = (uiPackedRGBA >> 16) & 0xff;
    rgba.w = (uiPackedRGBA >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba)
{
    unsigned int uiPackedRGBA = 0U;
    uiPackedRGBA |= 0x000000FF & (unsigned int)rgba.x;
    uiPackedRGBA |= 0x0000FF00 & (((unsigned int)rgba.y) << 8);
    uiPackedRGBA |= 0x00FF0000 & (((unsigned int)rgba.z) << 16);
    uiPackedRGBA |= 0xFF000000 & (((unsigned int)rgba.w) << 24);
    return uiPackedRGBA;
}

// Transpose kernel (see transpose SDK sample for details)
//*****************************************************************
__global__ 
void Transpose(const unsigned int* uiDataIn, 
                     unsigned int* uiDataOut, 
               const size_t iWidth, const size_t iHeight)
{
    // read the matrix tile into LMEM
    size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ unsigned int uiLocalBuff[16*17];

    if((xIndex < iWidth) && (yIndex < iHeight))
    {
        //uiLocalBuff[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = uiDataIn[(yIndex * iWidth) + xIndex];
        uiLocalBuff[threadIdx.y * (blockDim.x + 1) + threadIdx.x] = uiDataIn[(yIndex * iWidth) + xIndex];
    }

    // Synchronize the read into LMEM
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    if((xIndex < iHeight) && (yIndex < iWidth))
    {
        uiDataOut[(yIndex * iHeight) + xIndex] = 
         //uiLocalBuff[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)];
         uiLocalBuff[threadIdx.x * (blockDim.y + 1) + threadIdx.y];
    }
}

// 	simple 1st order recursive filter kernel 
//*****************************************************************
//    - processes one image column per thread
//      parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a  - blur parameter
//*****************************************************************
__global__ void SimpleRecursiveRGBA(
  const unsigned int* uiDataIn,
        unsigned int* uiDataOut,
  const size_t iWidth, const size_t iHeight, const float a)
{
    // compute X pixel location and check in-bounds
  size_t X = blockIdx.x * blockDim.x + threadIdx.x;
  if (X >= iWidth) return;
    
  // advance global pointers to correct column for this work item and x position
  uiDataIn += X;    
  uiDataOut += X;

  // start forward filter pass
  float4 yp = rgbaUintToFloat4(*uiDataIn);  // previous output
  for (int Y = 0; Y < iHeight; Y++) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);   
    *uiDataOut = rgbaFloat4ToUint(yc);
    yp = yc;
    uiDataIn += iWidth;     // move to next row
    uiDataOut += iWidth;    // move to next row
  }

  // reset global pointers to point to last element in column for this work item and x position
  uiDataIn -= iWidth;
  uiDataOut -= iWidth;

  // start reverse filter pass: ensures response is symmetrical
  yp = rgbaUintToFloat4(*uiDataIn);
  for (int Y = iHeight - 1; Y > -1; Y--) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);
    *uiDataOut = rgbaFloat4ToUint((rgbaUintToFloat4(*uiDataOut) + yc) * 0.5f);
    yp = yc;
    uiDataIn -= iWidth;   // move to previous row
    uiDataOut -= iWidth;  // move to previous row
  }
}

// Recursive Gaussian filter 
//*****************************************************************
//  parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//
//      If used, CLAMP_TO_EDGE is passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
__global__ void RecursiveRGBA(
  const unsigned int* uiDataIn, 
  unsigned int* uiDataOut, 
  const size_t iWidth, const size_t iHeight, 
  const float a0, const float a1, 
  const float a2, const float a3, 
  const float b1, const float b2, 
  const float coefp, const float coefn)
{
    // compute X pixel location and check in-bounds
    size_t X = blockIdx.x * blockDim.x + threadIdx.x;
    if (X >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    uiDataIn += X;    
    uiDataOut += X;

    // start forward filter pass
    float4 xp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous input
    float4 yp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output
    float4 yb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output by 2

#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*uiDataIn); 
    yb = xp * make_float4(coefp,coefp,coefp,coefp); 
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
		*uiDataOut = rgbaFloat4ToUint(yc);
        xp = xc; 
        yb = yp; 
        yp = yc; 
        uiDataIn += iWidth;     // move to next row
        uiDataOut += iWidth;    // move to next row
    }

    // reset global pointers to point to last element in column for this work item and x position
    uiDataIn -= iWidth;
    uiDataOut -= iWidth;

    // start reverse filter pass: ensures response is symmetrical
    float4 xn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 yn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ya = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*uiDataIn);
    xa = xn; 
    yn = xn * make_float4(coefn,coefn,coefn,coefn); 
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y > -1; Y--) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
        *uiDataOut = rgbaFloat4ToUint(rgbaUintToFloat4(*uiDataOut) + yc);
        uiDataIn -= iWidth;   // move to previous row
        uiDataOut -= iWidth;  // move to previous row
    }
}


#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include "study_cases.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
const int G=16;const unsigned GUARD=0xdeadbeef;
template<class T>std::vector<T>readv(const char*n){std::ifstream f(std::string("/home/imo/HeCBench_julia/src/recursiveGaussian-julia/")+n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixtureopen");size_t sz=f.tellg();f.seekg(0);std::vector<T>v(sz/sizeof(T));if(!f.read((char*)v.data(),sz))throw std::runtime_error("fixtureread");return v;}
struct State{Case c;GaussParms gp;unsigned*in,*tmp,*out;std::vector<unsigned>host;};
void rec(State&a,unsigned*in,unsigned*out,int w,int h){auto g=a.gp;RecursiveRGBA<<<(w+255)/256,256>>>(in+G,out+G,w,h,g.a0,g.a1,g.a2,g.a3,g.b1,g.b2,g.coefp,g.coefn);}
void trans(unsigned*in,unsigned*out,int w,int h){Transpose<<<dim3((w+15)/16,(h+15)/16),dim3(16,16)>>>(in+G,out+G,w,h);}
void seq(State&a){auto c=a.c;rec(a,a.in,a.tmp,c.w,c.h);trans(a.tmp,a.out,c.w,c.h);rec(a,a.out,a.tmp,c.h,c.w);trans(a.tmp,a.out,c.h,c.w);}
std::vector<unsigned>get(unsigned*p,int n){std::vector<unsigned>v(n+32);CK(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));return v;}
void encode(std::vector<unsigned>&art,const std::vector<unsigned>&v){for(auto p:v)for(int j=0;j<4;j++)art.push_back((p>>(8*j))&255);}
int main(){auto inputs=readv<unsigned>("inputs_u32.bin");auto gold=readv<unsigned>("oracle_u32.bin");auto coeff=readv<float>("coefficients_f32.bin");std::vector<State>states;std::vector<unsigned>prefix;for(auto c:cases){State a;a.c=c;PreProcessGaussParms(c.sigma,0,&a.gp);if(memcmp(&a.gp,coeff.data()+c.co,48))throw std::runtime_error("sourcecoeffbytesdiffer");int n=c.w*c.h;a.host.assign(n+32,GUARD);std::copy(inputs.begin()+c.io,inputs.begin()+c.io+n,a.host.begin()+G);std::vector<unsigned>sentinel(n+32,GUARD);CK(cudaMalloc(&a.in,(n+32)*4));CK(cudaMalloc(&a.tmp,(n+32)*4));CK(cudaMalloc(&a.out,(n+32)*4));CK(cudaMemcpy(a.in,a.host.data(),(n+32)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(a.tmp,sentinel.data(),(n+32)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(a.out,sentinel.data(),(n+32)*4,cudaMemcpyHostToDevice));for(auto p:{a.in,a.tmp,a.out}){auto v=get(p,n);prefix.insert(prefix.end(),v.begin(),v.end());}unsigned words[12];memcpy(words,&a.gp,48);prefix.insert(prefix.end(),words,words+12);seq(a);CK(cudaDeviceSynchronize());states.push_back(a);}
for(int trial=1;trial<=5;trial++){std::vector<unsigned>art=prefix;bool ok=true;int maxerr=0;double total=0;for(auto&a:states){auto c=a.c;int n=c.w*c.h;std::vector<unsigned>sentinel(n+32,GUARD);CK(cudaMemcpy(a.tmp,sentinel.data(),(n+32)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(a.out,sentinel.data(),(n+32)*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<3;r++)seq(a);CK(cudaDeviceSynchronize());auto z=std::chrono::steady_clock::now();total+=std::chrono::duration<double>(z-t).count()/3;auto timedtmp=get(a.tmp,n),timedout=get(a.out,n);std::vector<std::vector<unsigned>>stages;rec(a,a.in,a.tmp,c.w,c.h);CK(cudaDeviceSynchronize());stages.push_back(get(a.tmp,n));trans(a.tmp,a.out,c.w,c.h);CK(cudaDeviceSynchronize());stages.push_back(get(a.out,n));rec(a,a.out,a.tmp,c.h,c.w);CK(cudaDeviceSynchronize());stages.push_back(get(a.tmp,n));trans(a.tmp,a.out,c.h,c.w);CK(cudaDeviceSynchronize());stages.push_back(get(a.out,n));ok&=timedtmp==stages[2]&&timedout==stages[3];stages.push_back(timedtmp);stages.push_back(timedout);auto in=get(a.in,n);ok&=in==a.host;art.insert(art.end(),in.begin(),in.end());int refs[6]={0,1,2,3,2,3};for(int k=0;k<6;k++){auto&v=stages[k];for(int i=0;i<n+32;i++)if(i<G||i>=n+G)ok&=v[i]==GUARD;else{auto ref=gold[c.oo+refs[k]*n+i-G];for(int j=0;j<4;j++){int err=abs(int((v[i]>>(8*j))&255)-int((ref>>(8*j))&255));ok&=err<=2;maxerr=std::max(maxerr,err);if(c.mode==2)ok&=v[i]==0;}}encode(art,v);}}
std::string path="/tmp/recursiveGaussian_"+std::to_string(getpid())+"_"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)art.data(),art.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_five_fourkernel_batches_R3_plus_completion_mean_per_pipeline\",\"max_abs_error\":%d,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",total,maxerr,path.c_str(),art.size(),art.size());}}
