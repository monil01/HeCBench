#define main original_hecbench_main
#include "../jacobian-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
int dim=argc>2?atoi(argv[2]):128;bool k2=argc>1&&std::string(argv[1])=="k2";int batch=8,repeats=10;size_t n=(size_t)batch*dim*dim;std::vector<float>host(batch*dim),got(n),source_ref(n);std::vector<double>expected(n);for(size_t i=0;i<host.size();i++)host[i]=(float)((int)((i*37+11)%193)-96)/32.f;reference(host.data(),source_ref.data(),batch,dim);for(int b=0;b<batch;b++){std::vector<double>y(dim);double maximum=-INFINITY,sum=0;for(int i=0;i<dim;i++)maximum=fmax(maximum,host[b*dim+i]);for(int i=0;i<dim;i++){y[i]=exp((double)host[b*dim+i]-maximum);sum+=y[i];}for(double&v:y)v/=sum;for(int i=0;i<dim;i++)for(int j=0;j<dim;j++)expected[(size_t)b*dim*dim+i*dim+j]=y[i]*((i==j?1.:0.)-y[j]);}float *input,*out;cudaMalloc(&input,host.size()*4);cudaMalloc(&out,n*4);cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice);
 auto launch=[&](){if(k2)softmax_jacobian_k2<float><<<batch,256,(dim+256)*4>>>(input,out,dim);else softmax_jacobian_k1<float><<<batch,256,(dim+256)*4>>>(input,out,dim);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);double source_error=0,rowerr=0,sym=0,minimum_quad=INFINITY;bool invariant=true;for(size_t i=0;i<n;i++)source_error=fmax(source_error,fabs((double)got[i]-source_ref[i]));for(int b=0;b<batch;b++){double quad=0;for(int i=0;i<dim;i++){double rowsum=0,vi=i%7-3;for(int j=0;j<dim;j++){float value=got[(size_t)b*dim*dim+i*dim+j];rowsum+=value;sym=fmax(sym,fabs((double)value-got[(size_t)b*dim*dim+j*dim+i]));quad+=vi*value*(j%7-3);invariant&=i==j?value>0:value<0;}rowerr=fmax(rowerr,fabs(rowsum));}minimum_quad=fmin(minimum_quad,quad);}invariant&=source_error<1e-3&&rowerr<1e-6&&sym<1e-6&&minimum_quad>=-1e-6;if(!invariant)return 6;printf("STUDY_INVARIANTS originalCPUerror=%.12g maxrowsum=%.12g maxsymmetry=%.12g minprobequadratic=%.12g\n",source_error,rowerr,sym,minimum_quad);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-06));ok&=std::isfinite((double)got[j])&&e<=1e-06;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
