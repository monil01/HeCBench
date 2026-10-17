#define main original_hecbench_main
#include "../damage-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=262144,m=n/256,repeats=10;std::vector<int>input(n),family(m),count(m);std::vector<double>damage(m),got(2*m),expected(2*m);
for(int i=0;i<n;i++)input[i]=((17*i+13*(i/256))%23>=(i/256)%11)?i%m:-1;
for(int g=0;g<m;g++){int sum=0;for(int j=0;j<256;j++)sum+=input[g*256+j]!=-1;family[g]=sum+1+(13*g)%64;expected[g]=sum;expected[m+g]=1.0-(double)sum/family[g];}
int *di,*df,*dc;double *dd;cudaMalloc(&di,n*4);cudaMalloc(&df,m*4);cudaMalloc(&dc,m*4);cudaMalloc(&dd,m*8);cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(df,family.data(),m*4,cudaMemcpyHostToDevice);

 auto launch=[&](){damage_of_node<<<m,256,1024>>>(n,di,df,dc,dd);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(dc,0,m*4);cudaMemset(dd,0,m*8);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(count.data(),dc,m*4,cudaMemcpyDeviceToHost);cudaMemcpy(damage.data(),dd,m*8,cudaMemcpyDeviceToHost);for(int i=0;i<m;i++){got[i]=count[i];got[m+i]=damage[i];}
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-06));ok&=std::isfinite((double)got[j])&&e<=1e-06;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(double),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
