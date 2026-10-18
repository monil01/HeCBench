#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include "../simpleSpmv-cuda/kernels.cu"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<class T>std::vector<T> read(std::ifstream&f,size_t n){std::vector<T>a(n);f.read((char*)a.data(),n*sizeof(T));if(!f)exit(3);return a;}
template<class T>T* upload(const std::vector<T>&a){T*d;CK(cudaMalloc(&d,a.size()*sizeof(T)));CK(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>void capture(std::vector<double>&out,T*d,size_t n){std::vector<T>h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));for(auto v:h)out.push_back((double)v);}
struct Case{size_t n,nz;int tpr;float *mat,*x,*val;size_t*row,*col;};
void launch(Case&c,float*y,int bs,int method){if(method==0)mv_dense<<<(c.n+bs-1)/bs,bs>>>(c.n,c.mat,c.x,y);else if(method==1)mv_csr<<<(c.n+bs-1)/bs,bs>>>(c.n,c.row,c.col,c.val,c.x,y);else{dim3 block(c.tpr,bs/c.tpr);dim3 grid((c.n+block.y-1)/block.y);
#define V(B) vector_mv_csr<B><<<grid,block>>>(c.n,c.row,c.col,c.val,c.x,y)
switch(c.tpr){case 2:V(2);break;case 4:V(4);break;case 8:V(8);break;case 16:V(16);break;case 32:V(32);break;} // Replaced below with explicit source specializations.
}CK(cudaGetLastError());}
int main(){std::ifstream f("fixture.bin",std::ios::binary);int nc=read<uint32_t>(f,1)[0];std::vector<Case>cases;std::vector<double>prefix;for(int i=0;i<nc;i++){auto h=read<uint32_t>(f,3);size_t n=h[0],nz=h[1];auto mat=read<float>(f,n*n),x=read<float>(f,n);auto row=read<size_t>(f,n+1),col=read<size_t>(f,nz);auto val=read<float>(f,nz);Case c{n,nz,(int)h[2],upload(mat),upload(x),upload(val),upload(row),upload(col)};capture(prefix,c.mat,n*n);capture(prefix,c.x,n);capture(prefix,c.row,n+1);capture(prefix,c.col,nz);capture(prefix,c.val,nz);cases.push_back(c);}
std::ifstream ef("expected_f64.bin",std::ios::binary);ef.seekg(0,std::ios::end);size_t es=ef.tellg();ef.seekg(0);auto expected=read<double>(ef,es/8);std::vector<float*>outs;for(auto&c:cases)for(int bs=32;bs<=1024;bs*=2)for(int m=0;m<3;m++){std::vector<float>v(c.n+16,16777215.f);auto y=upload(v);outs.push_back(y);launch(c,y,bs,m);}CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<double>data=prefix,times;int k=0;double total=0;for(auto&c:cases)for(int bs=32;bs<=1024;bs*=2)for(int m=0;m<3;m++){CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int r=0;r<100;r++)launch(c,outs[k],bs,m);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/100.;total+=dt;times.push_back(dt);capture(data,outs[k],c.n+16);k++;}bool ok=true;double err=0;size_t mismatches=0;for(size_t j=0;j<expected.size();j++){double v=data[prefix.size()+j],e=expected[j];bool good=std::isfinite(v)&&fabs(v-e)<=1e-5+1e-5*fabs(e);ok&=good;mismatches+=!good;err=fmax(err,fabs(v-e));}all&=ok;const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/simpleSpmv_trial%d.bin",dir?dir:".",trial);std::ofstream of(file,std::ios::binary);of.write((char*)data.data(),data.size()*8);of.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%lu,\"mismatches\":%lu,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"jit_excluded\":true,\"component_seconds\":[",trial,total,ok?"true":"false",expected.size(),mismatches,err);for(size_t j=0;j<times.size();j++)printf("%s%.12g",j?",":"",times[j]);printf("],\"timing_region\":\"sum90_source100repeat_kernelmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%lu,\"byte_order\":\"little\"}}\n",file,data.size());fflush(stdout);}puts(all?"PASS":"FAIL");return all?0:1;}
