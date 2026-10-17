#define main original_hecbench_main
#include "../topk-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int batch=128,hidden=4096,k=1024,repeats=10,n=batch*hidden,outsize=batch*k;std::vector<float>input(n),values(outsize),got(2*outsize),expected(2*outsize);std::vector<int>ids(outsize);for(int row=0;row<batch;row++){std::vector<std::pair<float,int>>ordered(hidden);for(int col=0;col<hidden;col++){float value=(5*col+17*row)%hidden;input[row*hidden+col]=value;ordered[col]={value,col};}std::partial_sort(ordered.begin(),ordered.begin()+k,ordered.end(),[](auto a,auto b){return a.first>b.first;});for(int rank=0;rank<k;rank++){expected[row*k+rank]=ordered[rank].first;expected[outsize+row*k+rank]=(float)ordered[rank].second;}}float*di,*dv;int*did;if(cudaMalloc(&di,n*4)!=cudaSuccess||cudaMalloc(&dv,outsize*4)!=cudaSuccess||cudaMalloc(&did,outsize*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){topk_permutation_rows<<<2048,256>>>(di,did,dv,batch,hidden,k);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(dv,0xff,outsize*4);cudaMemset(did,0xff,outsize*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(values.data(),dv,outsize*4,cudaMemcpyDeviceToHost);cudaMemcpy(ids.data(),did,outsize*4,cudaMemcpyDeviceToHost);for(int i=0;i<outsize;i++){got[i]=values[i];got[outsize+i]=(float)ids[i];}
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
