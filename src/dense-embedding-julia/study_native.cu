#define main original_hecbench_main
#include "../dense-embedding-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int mode=argc>1?atoi(argv[1]+1):1;constexpr int rows=512,batch=16,dim=768,repeats=10,count=rows*dim;std::vector<float>input(count),dense(batch*dim),got(count),expected(count,0.f);std::vector<int>offsets(batch+1,0);for(int i=0;i<count;i++)input[i]=(float)((17*i)%257-128)/128.f;for(int i=0;i<batch*dim;i++)dense[i]=(float)((13*i)%127-63)/64.f;for(int i=0;i<batch;i++)offsets[i+1]=offsets[i]+(2+i%7)*dim;reference(input.data(),dense.data(),expected.data(),dim,batch,offsets.data());float*di,*dd,*out;int*doff;if(cudaMalloc(&di,count*4)!=cudaSuccess||cudaMalloc(&dd,batch*dim*4)!=cudaSuccess||cudaMalloc(&out,count*4)!=cudaSuccess||cudaMalloc(&doff,(batch+1)*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),count*4,cudaMemcpyHostToDevice);cudaMemcpy(dd,dense.data(),batch*dim*4,cudaMemcpyHostToDevice);cudaMemcpy(doff,offsets.data(),(batch+1)*4,cudaMemcpyHostToDevice);
 auto launch=[&](){if(mode==1)dense_esuhm<float><<<batch,256>>>(di,dd,out,dim,doff);else if(mode==2)dense_esuhm2<float><<<batch,256>>>(di,dd,out,dim,doff);else dense_esuhm3<float><<<batch,256>>>(di,dd,out,dim,doff);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,count*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,count*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
