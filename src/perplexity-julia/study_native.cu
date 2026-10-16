#define main original_hecbench_main
#include "../perplexity-cuda/main.cu"
#undef main
int main(){
 constexpr int n=4096,k=32,repeats=10,epochs=100,count=n*k;constexpr float p=8.f,tol=1e-8f;
 std::vector<float>input(count),got(count),expected(count);for(int i=0;i<count;i++)input[i]=(float)((17*i+(i/32)*13)%257)/256.f;
 reference(input.data(),expected.data(),p,epochs,tol,n,k);
 float *di,*out;if(cudaMalloc(&di,count*4)!=cudaSuccess||cudaMalloc(&out,count*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),count*4,cudaMemcpyHostToDevice);
 const float entropy=logf(p);auto launch=[&](){sigmas_kernel<int,float><<<16,256>>>(di,out,p,entropy,epochs,tol,n,k);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,count*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,count*4,cudaMemcpyDeviceToHost);bool ok=status==cudaSuccess;double error=0,relative=0;for(int j=0;j<count;j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs(expected[j]),1e-3));ok&=std::isfinite(got[j])&&e<=1e-3;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),4,count,f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",count,error,relative,filename,count);
 }cudaFree(di);cudaFree(out);return allok?0:1;
}
