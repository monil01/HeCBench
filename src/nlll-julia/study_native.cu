#define main original_hecbench_main
#include "../nlll-cuda/main.cu"
#undef main
#include <vector>
int main(){
 constexpr int nf=4096,k=64,repeats=10;std::vector<float>input(nf*k),weights(nf);std::vector<int>target(nf);
 for(size_t i=0;i<input.size();i++)input[i]=-(float)((17*i)%257)/128.f;
 for(int i=0;i<nf;i++){weights[i]=(float)(i%7+1)/8.f;target[i]=i%k;}
 float loss=0,weight=0;for(int i=0;i<nf;i++){int t=target[i];if(t!=32){float w=weights[t];loss-=input[i*k+t]*w;weight+=w;}}float expected[2]={loss/weight,weight};
 float *di,*dw,*out,*total;int*dt;
 if(cudaMalloc(&di,input.size()*4)!=cudaSuccess||cudaMalloc(&dw,nf*4)!=cudaSuccess||cudaMalloc(&dt,nf*4)!=cudaSuccess||cudaMalloc(&out,4)!=cudaSuccess||cudaMalloc(&total,4)!=cudaSuccess)return 3;
 cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dw,weights.data(),nf*4,cudaMemcpyHostToDevice);cudaMemcpy(dt,target.data(),nf*4,cudaMemcpyHostToDevice);
 auto launch=[&](){nll_loss_forward_reduce2d_kernel<float,float,int,256><<<1,256>>>(out,total,di,dt,dw,true,nf,k,32);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,4);cudaMemset(total,0,4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 float got[2];cudaMemcpy(got,out,4,cudaMemcpyDeviceToHost);cudaMemcpy(got+1,total,4,cudaMemcpyDeviceToHost);bool ok=status==cudaSuccess;double error=0,relative=0;for(int j=0;j<2;j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs(expected[j]),1e-5));ok&=std::isfinite(got[j])&&e<=1e-5;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got,4,2,f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":2,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":2,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",error,relative,filename);
 }cudaFree(di);cudaFree(dw);cudaFree(dt);cudaFree(out);cudaFree(total);return allok?0:1;
}
