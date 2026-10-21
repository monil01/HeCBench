#include <complex>
#define main original_hecbench_main
#include "../chemv-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=370,repeats=1;std::vector<ComplexFloat> at(n*n),x(n),y(n),gotcomplex(n);std::vector<float> got(2*n),expected(2*n);
for(int col=0;col<n;col++)for(int row=0;row<n;row++)at[col*n+row]={(float)((7*col+11*row)%61-30)/2048.f,(float)((3*col+5*row)%59-29)/2048.f};
for(int i=0;i<n;i++){x[i]={(float)(i%31-15)/128.f,(float)((7*i)%29-14)/128.f};y[i]={(float)(i%19-9)/256.f,(float)((5*i)%17-8)/256.f};}
using C=std::complex<double>;C alpha(3.14f,1.59f),beta(2.71f,8.28f);
for(int row=0;row<n;row++){C sum(0);for(int col=0;col<n;col++){auto a=row>=col?at[col*n+row]:at[row*n+col];C val=row==col?C(a.Re,0):row>col?C(a.Re,a.Im):C(a.Re,-a.Im);sum+=val*C(x[col].Re,x[col].Im);}C v=alpha*sum+beta*C(y[row].Re,y[row].Im);expected[2*row]=v.real();expected[2*row+1]=v.imag();}
ComplexFloat *da,*dx,*dy;cudaMalloc(&da,n*n*sizeof(ComplexFloat));cudaMalloc(&dx,n*sizeof(ComplexFloat));cudaMalloc(&dy,n*sizeof(ComplexFloat));cudaMemcpy(da,at.data(),n*n*sizeof(ComplexFloat),cudaMemcpyHostToDevice);cudaMemcpy(dx,x.data(),n*sizeof(ComplexFloat),cudaMemcpyHostToDevice);cudaMemcpy(dy,y.data(),n*sizeof(ComplexFloat),cudaMemcpyHostToDevice);

 auto launch=[&](){chemv_kernel0<<<12,32>>>(da,dx,dy,1.59f,3.14f,8.28f,2.71f);chemv_kernel1<<<12,32>>>(da,dx,dy,1.59f,3.14f);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemcpy(dy,y.data(),n*sizeof(ComplexFloat),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(gotcomplex.data(),dy,n*sizeof(ComplexFloat),cudaMemcpyDeviceToHost);for(int i=0;i<n;i++){got[2*i]=gotcomplex[i].Re;got[2*i+1]=gotcomplex[i].Im;}
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-05));ok&=std::isfinite((double)got[j])&&e<=1e-05;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
