#include <vector>
#include <cmath>
#define main hecbench_original_main
#include "scan-cuda/main.cu"
#undef main
void emit(int trial,double seconds,const std::vector<double>&result,const std::vector<double>&expected,int repeat,double atol=0.,double rtol=0.){
 double error=0;bool correct=result.size()==expected.size();for(size_t i=0;i<result.size();i++){double e=fabs(result[i]-expected[i]);error=fmax(error,e);correct &= std::isfinite(result[i])&&e<=atol+rtol*fabs(expected[i]);}char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*f=fopen(path,"wb");fwrite(result.data(),8,result.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"timing_region\":\"synchronized_kernel_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,correct?"true":"false",result.size(),error,repeat,path,result.size());if(!correct)exit(1);
}
template<class T,int N>double study(int n,int repeat,std::vector<double>&values,std::vector<double>&expected){std::vector<T>input(n),out(n),ref(n);for(int i=0;i<n;i++)input[i]=T((i*17)%5+1);for(int base=0;base<n;base+=N){T acc=0;for(int j=0;j<N;j++){ref[base+j]=acc;acc+=input[base+j];}}T *di,*dr;cudaMalloc(&di,n*sizeof(T));cudaMalloc(&dr,n*sizeof(T));cudaMemcpy(di,input.data(),n*sizeof(T),cudaMemcpyHostToDevice);double total=0;for(int mode=0;mode<2;mode++){auto start=std::chrono::steady_clock::now();for(int j=0;j<repeat;j++){if(mode==0)scan<T,N><<<1280,N/2>>>(n/N,dr,di);else scan_bcao<T,N><<<1280,N/2>>>(n/N,dr,di);}cudaDeviceSynchronize();total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;cudaMemcpy(out.data(),dr,n*sizeof(T),cudaMemcpyDeviceToHost);for(T v:out)values.push_back(double(v));for(T v:ref)expected.push_back(double(v));}cudaFree(di);cudaFree(dr);return total;}
#define RUN(N) total+=study<char,N>(n,repeat,values,expected);total+=study<short,N>(n,repeat,values,expected);total+=study<int,N>(n,repeat,values,expected);total+=study<long,N>(n,repeat,values,expected);
void run(int n,int repeat,int trial){std::vector<double>values,expected;double total=0;RUN(128)RUN(256)RUN(512)RUN(1024)RUN(2048)if(trial)emit(trial,total,values,expected,repeat);}
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):3;run(n,1,0);for(int trial=1;trial<=5;trial++)run(n,repeat,trial);return 0;}
