#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../degrid-cuda/degrid.h"
#undef NPOINTS
#define NPOINTS 64
#undef IMG_SIZE
#define IMG_SIZE 64
#undef GCF_DIM
#define GCF_DIM 32
#undef REPEAT
#define REPEAT 3
#include "../degrid-cuda/kernels.cu"
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Case{std::vector<double2>points,image,gcf;std::vector<double>want;double2*dp;double2*di;double2*dg;double2*out;};
void encode(std::vector<double>&out,const std::vector<double2>&x){for(auto v:x){out.push_back(v.x);out.push_back(v.y);}}
int main(){std::vector<Case>cases;std::vector<double>prefix;int bases[9]={-1,0,1,15,32,62,63,64,47};
 for(int id=1;id<=2;++id){Case c;for(int i=0;i<64;++i)c.points.push_back(make_double2(bases[i%9]+double(i%8)/8,bases[(i*5)%9]+double(i/8)/8));auto key=[](double2 p){return int(floor(8*(p.y-floor(p.y))))+8*int(floor(8*(p.x-floor(p.x))));};std::stable_sort(c.points.begin(),c.points.end(),[&](double2 a,double2 b){return key(a)<key(b);});c.image.assign(8256,make_double2(0,0));for(int y=0;y<64;++y)for(int x=0;x<64;++x)c.image[2080+y*64+x]=make_double2(double((x*3+y*5)%31-15)/16,double((x*7+y*2)%29-14)/16);c.gcf.resize(65536);
  for(int sy=0;sy<8;++sy)for(int sx=0;sx<8;++sx)for(int y=0;y<32;++y)for(int x=0;x<32;++x)c.gcf[(sy*8+sx)*1024+y*32+x]=id==2?make_double2(x==16&&y==16?1:0,0):make_double2(double((x*3+y*5+sx*7+sy*11)%31-15)/64,double((x*7+y*3+sx*5+sy*2)%29-14)/64);
  for(auto p:c.points){int mx=int(floor(p.x)),my=int(floor(p.y)),sx=int(floor(8*(p.x-mx))),sy=int(floor(8*(p.y-my)));int64_t rr=0,ii=0;for(int dy=-16;dy<16;++dy)for(int dx=-16;dx<16;++dx){int x=mx+dx,y=my+dy;if(x<0||y<0||x>=64||y>=64)continue;auto v=c.image[2080+y*64+x],g=c.gcf[(sy*8+sx)*1024+(dy+16)*32+dx+16];int64_t vr=llround(v.x*16),vi=llround(v.y*16),gr=llround(g.x*64),gi=llround(g.y*64);rr+=vr*gr-vi*gi;ii+=vr*gi+vi*gr;}c.want.push_back(double(rr)/1024);c.want.push_back(double(ii)/1024);}
  check(cudaMalloc(&c.dp,64*16));check(cudaMalloc(&c.di,8256*16));check(cudaMalloc(&c.dg,65536*16));check(cudaMalloc(&c.out,64*16));check(cudaMemcpy(c.dp,c.points.data(),64*16,cudaMemcpyHostToDevice));check(cudaMemcpy(c.di,c.image.data(),8256*16,cudaMemcpyHostToDevice));check(cudaMemcpy(c.dg,c.gcf.data(),65536*16,cudaMemcpyHostToDevice));encode(prefix,c.points);encode(prefix,c.image);encode(prefix,c.gcf);prefix.insert(prefix.end(),128,0);for(double v:{64.,64.,32.,8.,2080.,3.,double(id)})prefix.push_back(v);cases.push_back(std::move(c));
 }
 auto run=[&](Case&c,std::vector<double>&out,double&error){check(cudaMemset(c.out,0,64*16));check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int i=0;i<3;++i)degrid_kernel<<<2,dim3(32,8)>>>(c.out,c.dp,64,c.di+2080,64,c.dg+528);check(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/3;std::vector<double2>got(64);check(cudaMemcpy(got.data(),c.out,64*16,cudaMemcpyDeviceToHost));std::vector<double>encoded;encode(encoded,got);for(size_t i=0;i<encoded.size();++i)error=std::max(error,fabs(encoded[i]-c.want[i]));out.insert(out.end(),encoded.begin(),encoded.end());return elapsed;};
 for(auto&c:cases){std::vector<double>out;double err=0;run(c,out,err);if(err>1e-7)return 3;}
 for(int trial=1;trial<=5;++trial){auto out=prefix;double elapsed=0,error=0;for(auto&c:cases)elapsed+=run(c,out,error);std::string name="trial_"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();bool ok=error<=1e-7;std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<error<<",\"timing_region\":\"sum_two_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
 for(auto&c:cases){cudaFree(c.dp);cudaFree(c.di);cudaFree(c.dg);cudaFree(c.out);}}
