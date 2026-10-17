#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <tuple>
#include "../hogbom-cuda/kernels.h"
unsigned int niters=19;
const float gain=.1f;
const float threshold=.00001f;
#include "../hogbom-cuda/kernels.cu"
#include "../hogbom-cuda/timer.cpp"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Case{int w;std::vector<float>dirty,psf,model;std::vector<double>want,trace;float*dr;float*dp;Peak*pk;};
auto priority(size_t i){return std::make_pair(i%32768,i/32768);}
std::vector<double> oracle(const Case&c){auto res=c.dirty;std::vector<float>model(res.size());std::vector<double>trace;
 for(int it=0;it<19;++it){size_t best=0;float val=0;
  for(size_t i=0;i<res.size();++i)if(fabsf(res[i])>fabsf(val)||(fabsf(res[i])==fabsf(val)&&priority(i)<priority(best))){best=i;val=res[i];}
  if(fabsf(val)<=threshold)throw std::runtime_error("threshold active");trace.push_back(best);trace.push_back(val);float factor=gain*val;model[best]+=factor;int sx=best%c.w,sy=best/c.w;
  for(int y=std::max(0,sy-3);y<=std::min(c.w-1,sy+3);++y)for(int x=std::max(0,sx-3);x<=std::min(c.w-1,sx+3);++x)res[y*c.w+x]=std::fma(-factor,c.psf[(y-sy+3)*7+x-sx+3],res[y*c.w+x]);
 }
 std::vector<double>out(model.begin(),model.end());out.insert(out.end(),res.begin(),res.end());out.push_back(24);out.push_back(1);out.insert(out.end(),trace.begin(),trace.end());return out;
}
int main(){std::vector<Case>cases;std::vector<double>prefix;int id=0;
 for(int w:{65,257}){Case c;c.w=w;c.psf.assign(49,0);c.dirty.assign(w*w,0);c.model.assign(w*w,0);c.trace.resize(38);
  for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)c.psf[(y+3)*7+x+3]=(!x&&!y)?1.f:(!y?.125f:(!x?-.0625f:.03125f));
  std::vector<std::tuple<int,int,float>>sources=w==65?std::vector<std::tuple<int,int,float>>{{0,0,4},{32,20,-3},{64,50,2}}:std::vector<std::tuple<int,int,float>>{{32769%w,32769/w,4},{300%w,300/w,4}};
  for(auto[sx,sy,a]:sources)for(int y=0;y<w;++y)for(int x=0;x<w;++x){int px=x-sx+3,py=y-sy+3;if(px>=0&&px<7&&py>=0&&py<7)c.dirty[y*w+x]+=a*c.psf[py*7+px];}
  prefix.insert(prefix.end(),c.dirty.begin(),c.dirty.end());prefix.insert(prefix.end(),c.psf.begin(),c.psf.end());prefix.insert(prefix.end(),c.model.begin(),c.model.end());for(double v:{double(w),7.,19.,double(gain),double(threshold),double(++id)})prefix.push_back(v);
  c.want=oracle(c);check(cudaMalloc(&c.dr,w*w*sizeof(float)));check(cudaMalloc(&c.dp,49*sizeof(float)));check(cudaMalloc(&c.pk,128*sizeof(Peak)));check(cudaMemcpy(c.dp,c.psf.data(),49*sizeof(float),cudaMemcpyHostToDevice));cases.push_back(std::move(c));
 }
 auto run=[&](Case&c,std::vector<double>&out,double&error){check(cudaMemcpy(c.dr,c.dirty.data(),c.dirty.size()*sizeof(float),cudaMemcpyHostToDevice));std::fill(c.model.begin(),c.model.end(),0.f);Peak psfpeak=findPeak(c.dp,c.pk,49);check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
  for(int it=0;it<19;++it){Peak p=findPeak(c.dr,c.pk,c.dirty.size());c.trace[2*it]=p.pos;c.trace[2*it+1]=p.val;subtractPSF(c.dp,7,c.dr,c.w,p.pos,psfpeak.pos,p.val,gain);c.model[p.pos]+=p.val*gain;}
  check(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<float>res(c.dirty.size());check(cudaMemcpy(res.data(),c.dr,res.size()*sizeof(float),cudaMemcpyDeviceToHost));std::vector<double>got(c.model.begin(),c.model.end());got.insert(got.end(),res.begin(),res.end());got.push_back(psfpeak.pos);got.push_back(psfpeak.val);got.insert(got.end(),c.trace.begin(),c.trace.end());for(size_t i=0;i<got.size();++i)error=std::max(error,fabs(got[i]-c.want[i]));for(int it=0;it<19;++it)if(c.trace[2*it]!=c.want[2*c.dirty.size()+2+2*it]||fabs(c.trace[2*it+1])<=threshold)error=1e9;out.insert(out.end(),got.begin(),got.end());return elapsed;
 };
 for(auto&c:cases){std::vector<double>out;double e=0;run(c,out,e);}
 for(int trial=1;trial<=5;++trial){auto out=prefix;double elapsed=0,error=0;for(auto&c:cases)elapsed+=run(c,out,error);std::string name="trial_"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*sizeof(double));f.close();bool ok=error<=.001;std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<error<<",\"timing_region\":\"sum_two_19_iteration_clean_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
 for(auto&c:cases){cudaFree(c.dr);cudaFree(c.dp);cudaFree(c.pk);}
}
