#define main hecbench_original_main
#include "relu-cuda/main.cu"
#undef main
#include <vector>
int pack_reference(unsigned x){unsigned out=0;for(unsigned s=0;s<32;s+=8){int v=(x>>s)&255;v=v>=128?v-256:v;out|=unsigned(v>0?v:0)<<s;}return int(out);}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):100;
 std::vector<half>feature(n),gradient(n),backprop(n);std::vector<int>input(n),output(n);
 for(int i=0;i<n;i++){feature[i]=__float2half(float(i%17-8)/8.f);gradient[i]=__float2half(1.f);input[i]=int(unsigned(i)*2654435761u);}
 half *df,*dg,*db;int *dx,*dy;cudaMalloc(&df,n*sizeof(half));cudaMalloc(&dg,n*sizeof(half));cudaMalloc(&db,n*sizeof(half));cudaMalloc(&dx,n*sizeof(int));cudaMalloc(&dy,n*sizeof(int));
 cudaMemcpy(df,feature.data(),n*sizeof(half),cudaMemcpyHostToDevice);cudaMemcpy(dg,gradient.data(),n*sizeof(half),cudaMemcpyHostToDevice);cudaMemcpy(dx,input.data(),n*sizeof(int),cudaMemcpyHostToDevice);
 for(int phase=0;phase<4;phase++){
  if(phase==0)ReluGrad_impl1<<<divup(divup(n,2),256),256>>>(dg,df,db,n);
  if(phase==1)ReluGrad_impl2<<<divup(divup(n,8),256),256>>>(dg,df,db,n);
  if(phase==2)Relu_impl1<<<divup(n,256),256>>>(n,dx,dy);
  if(phase==3)Relu_impl2<<<divup(n,256),256>>>(n,dx,dy);
 }cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[4],total=0;std::vector<double>values;bool correct=true;
  for(int phase=0;phase<4;phase++){
   cudaMemset(db,0,n*sizeof(half));cudaMemset(dy,0,n*sizeof(int));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
   for(int r=0;r<repeat;r++){
    if(phase==0)ReluGrad_impl1<<<divup(divup(n,2),256),256>>>(dg,df,db,n);
    if(phase==1)ReluGrad_impl2<<<divup(divup(n,8),256),256>>>(dg,df,db,n);
    if(phase==2)Relu_impl1<<<divup(n,256),256>>>(n,dx,dy);
    if(phase==3)Relu_impl2<<<divup(n,256),256>>>(n,dx,dy);
   }
   cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[phase]=std::chrono::duration<double>(stop-start).count()/repeat;total+=times[phase];
   if(phase<2){cudaMemcpy(backprop.data(),db,n*sizeof(half),cudaMemcpyDeviceToHost);for(int i=0;i<n;i++){float v=__half2float(backprop[i]);float want=__half2float(feature[i])>0?1.f:0.f;correct &= v==want;values.push_back(v);}}
   else{cudaMemcpy(output.data(),dy,n*sizeof(int),cudaMemcpyDeviceToHost);for(int i=0;i<n;i++){correct &= output[i]==pack_reference(unsigned(input[i]));values.push_back(double(unsigned(output[i])));}}
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(double),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"sum_four_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,total,correct?"true":"false",4*n,repeat,times[0],times[1],times[2],times[3],filename,4*n);
  if(!correct)return 1;
 }
 cudaFree(df);cudaFree(dg);cudaFree(db);cudaFree(dx);cudaFree(dy);return 0;
}
