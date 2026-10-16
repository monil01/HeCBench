#define main original_hecbench_main
#include "../logprob-cuda/main.cu"
#undef main
#include <vector>
int main(){
 constexpr int maxlen=32,batch=64,vocab=256,repeats=10;constexpr int count=batch*(maxlen-1)+batch;
 std::vector<float>input(batch*maxlen*vocab),lp(batch*(maxlen-1)),cum(batch),got(count),expected(count);std::vector<int>ids(batch*maxlen),lengths(batch,maxlen);
 for(size_t i=0;i<input.size();i++)input[i]=(float)((int)((17*i)%97)-48)/8.f;for(size_t i=0;i<ids.size();i++)ids[i]=(13*i)%256;
 log_probs_cpu<float>(lp.data(),input.data(),ids.data(),lengths.data(),maxlen,batch,vocab,vocab,cum.data());std::copy(lp.begin(),lp.end(),expected.begin());std::copy(cum.begin(),cum.end(),expected.begin()+lp.size());
 float *di,*out,*total;int *dids,*dl;
 if(cudaMalloc(&di,input.size()*4)!=cudaSuccess||cudaMalloc(&dids,ids.size()*4)!=cudaSuccess||cudaMalloc(&dl,batch*4)!=cudaSuccess||cudaMalloc(&out,lp.size()*4)!=cudaSuccess||cudaMalloc(&total,batch*4)!=cudaSuccess)return 3;
 cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dids,ids.data(),ids.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dl,lengths.data(),batch*4,cudaMemcpyHostToDevice);
 auto launch=[&](){log_probs_kernel<float><<<dim3(31,64),256>>>(out,di,dids,dl,maxlen,batch,vocab,vocab);accumulate_log_probs<<<batch,256>>>(total,out,dl,maxlen,batch);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,lp.size()*4);cudaMemset(total,0,batch*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,lp.size()*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+lp.size(),total,batch*4,cudaMemcpyDeviceToHost);bool ok=status==cudaSuccess;double error=0,relative=0;for(int j=0;j<count;j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs(expected[j]),1e-3));ok&=std::isfinite(got[j])&&e<=1e-3;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),4,count,f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",count,error,relative,filename,count);
 }cudaFree(di);cudaFree(dids);cudaFree(dl);cudaFree(out);cudaFree(total);return allok?0:1;
}
