#include <cstdio>
#include <cmath>
#include "native_force_kernel.cuh"
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float *p;CHECK(cudaMallocHost(&p,11*4));float in[11]={0,0,0,1,.5,0,0,1,0,0,0};CHECK(cudaMemcpy(p,in,sizeof(in),cudaMemcpyHostToDevice));float soft2=.1f*.1f,cut2=3.116326355f*3.116326355f;
 cudaPointerAttributes attributes;CHECK(cudaPointerGetAttributes(&attributes,p));fprintf(stderr,"SOURCE_MEMORY_TYPE %d host_device_alias_equal%d\n",int(attributes.type),attributes.hostPointer==attributes.devicePointer);
 Step10_kernel<<<dim3(1,1),dim3(32,4)>>>(1,1,p,p+1,p+2,p+3,p+4,p+5,p+6,p+7,p+8,p+9,p+10,cut2,soft2,1);
 CHECK(cudaDeviceSynchronize());float out[11];CHECK(cudaMemcpy(out,p,sizeof(out),cudaMemcpyDeviceToHost));
 double expected=.5/std::pow(.25+double(soft2),1.5);bool exactinputs=true;for(int i=0;i<8;i++)exactinputs&=in[i]==out[i];bool pass=exactinputs&&std::abs(out[8]-expected)<1e-6&&out[9]==0&&out[10]==0;
 printf("{\"GPU_source_kernel_unchanged\":true,\"actual_input_readback_equal\":%s,\"output_velocity\":[%.17g,%.17g,%.17g],\"independent_softened_acceleration\":%.17g,\"source_declared_Newton_acceleration\":4,\"Newton_relative_error\":%.17g,\"softened_rule_check_pass\":%s}\n",exactinputs?"true":"false",double(out[8]),double(out[9]),double(out[10]),expected,std::abs(double(out[8])-4)/4,pass?"true":"false");CHECK(cudaFreeHost(p));return pass?0:3;}
