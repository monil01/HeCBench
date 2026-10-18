#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <memory>
#define NX 16
#define NZ 8
#define SIM_TIME 1
#define DATA_SPEC DATA_SPEC_THERMAL
#define main historical_main
#include "../miniWeather-cuda/main.cu"
#undef main
const double SENT=-8192.25;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
void mpi_ck(int e){if(e){fprintf(stderr,"MPI error %d\n",e);exit(2);}}
std::vector<double>load(const char*f){std::ifstream in(f,std::ios::binary|std::ios::ate);size_t z=in.tellg();std::vector<double>a(z/8);in.seekg(0);in.read((char*)a.data(),z);return a;}
struct Buf{std::vector<double>h;double*raw;double*ptr;Buf(std::vector<double>v):h(v.size()+8,SENT){std::copy(v.begin(),v.end(),h.begin()+4);ck(cudaMalloc(&raw,h.size()*8));ptr=raw+4;reset();}void reset(){ck(cudaMemcpy(raw,h.data(),h.size()*8,cudaMemcpyHostToDevice));}std::vector<double>get(){std::vector<double>a(h.size());ck(cudaMemcpy(a.data(),raw,a.size()*8,cudaMemcpyDeviceToHost));return a;}bool guards(const std::vector<double>&a){for(int j=0;j<4;j++)if(a[j]!=SENT||a[a.size()-1-j]!=SENT)return false;return true;}};
struct Case{int nx,nz,steps;double dx,dz;std::vector<double>schedule;std::vector<Buf*>A,H;std::vector<std::vector<double>>host;Buf*mass;Buf*energy;};
void activate(Case&c){nx=c.nx;nz=c.nz;dx=c.dx;dz=c.dz;k_beg=i_beg=0;left_rank=right_rank=0;data_spec_int=DATA_SPEC_THERMAL;sendbuf_l=c.host[0].data();sendbuf_r=c.host[1].data();recvbuf_l=c.host[2].data();recvbuf_r=c.host[3].data();}
void reset(Case&c){for(auto*b:c.A)b->reset();for(auto&h:c.host)std::fill(h.begin(),h.end(),0.);direction_switch=1;ck(cudaDeviceSynchronize());}
void integrate(Case&c){activate(c);direction_switch=1;for(auto ds:c.schedule){dt=ds;perform_timestep(c.A[0]->ptr,c.A[1]->ptr,c.A[2]->ptr,c.A[3]->ptr,c.H[0]->ptr,c.H[1]->ptr,c.H[2]->ptr,c.H[3]->ptr,c.H[4]->ptr,c.A[4]->ptr,c.A[5]->ptr,c.A[6]->ptr,c.A[7]->ptr,dt);}ck(cudaDeviceSynchronize());}
std::vector<double>diag(Case&c){ck(cudaMemset(c.mass->ptr,0,8));ck(cudaMemset(c.energy->ptr,0,8));acc_mass_te<<<dim3((c.nx+15)/16,(c.nz+15)/16),dim3(16,16)>>>(c.mass->ptr,c.energy->ptr,c.A[0]->ptr,c.H[0]->ptr,c.H[1]->ptr,c.nx,c.nz,c.dx,c.dz);double loc[2],glob[2];ck(cudaMemcpy(loc,c.mass->ptr,8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(loc+1,c.energy->ptr,8,cudaMemcpyDeviceToHost));mpi_ck(MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD));return{glob[0],glob[1]};}
int main(int argc,char**argv){mpi_ck(MPI_Init(&argc,&argv));mpi_ck(MPI_Comm_size(MPI_COMM_WORLD,&nranks));mpi_ck(MPI_Comm_rank(MPI_COMM_WORLD,&myrank));if(nranks!=1||myrank!=0)return 2;auto f=load(MINI_FIXTURE),oracle=load(MINI_ORACLE);int nc=f[0];size_t pos=1;auto take=[&](size_t n){std::vector<double>a(f.begin()+pos,f.begin()+pos+n);pos+=n;return a;};std::vector<Case>cases;
for(int ci=0;ci<nc;ci++){Case c;c.nx=f[pos++];c.nz=f[pos++];c.steps=f[pos++];c.dx=f[pos++];c.dz=f[pos++];pos++;c.schedule=take(c.steps);auto initial=take(4*(c.nx+4)*(c.nz+4));c.A.push_back(new Buf(initial));c.A.push_back(new Buf(initial));for(int n:{4*(c.nx+1)*(c.nz+1),4*c.nx*c.nz,8*c.nz,8*c.nz,8*c.nz,8*c.nz})c.A.push_back(new Buf(std::vector<double>(n)));for(int n:{c.nz+4,c.nz+4,c.nz+1,c.nz+1,c.nz+1})c.H.push_back(new Buf(take(n)));c.host=std::vector<std::vector<double>>(4,std::vector<double>(8*c.nz));c.mass=new Buf({0.});c.energy=new Buf({0.});cases.push_back(c);}
for(auto&c:cases){reset(c);diag(c);integrate(c);diag(c);}
for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double total=0,maxerror=0;std::vector<double>times;size_t oi=0;
for(auto&c:cases){reset(c);auto d0=diag(c);ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();integrate(c);double tm=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();total+=tm;times.push_back(tm);auto d1=diag(c);
for(auto*b:c.H){auto a=b->get();ok&=b->guards(a)&&a==b->h;payload.insert(payload.end(),a.begin(),a.end());}
std::vector<std::vector<double>>values;for(auto*b:c.A){auto a=b->get();ok&=b->guards(a);payload.insert(payload.end(),a.begin(),a.end());values.emplace_back(a.begin()+4,a.end()-4);}std::vector<double>d=d0;d.insert(d.end(),d1.begin(),d1.end());payload.insert(payload.end(),d.begin(),d.end());values.push_back(d);
for(auto&a:values){for(auto x:a){double y=oracle[oi++],error=fabs(x-y);maxerror=std::max(maxerror,error);ok&=std::isfinite(x)&&error<=2e-8+2e-11*fabs(y);}}ok&=fabs(d1[0]-d0[0])<=1e-12*d0[0];for(auto*b:{c.mass,c.energy}){auto a=b->get();ok&=b->guards(a);payload.insert(payload.end(),a.begin(),a.end());}for(int i=0;i<4;i++){ok&=c.host[i]==values[i+4];payload.insert(payload.end(),c.host[i].begin(),c.host[i].end());}}
std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*8);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum3_full_splitRK_MPIself_halo_pipeline_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":162,\"case_times_seconds\":[",trial,total,ok?"true":"false",payload.size(),maxerror);for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok){MPI_Finalize();return 3;}}
mpi_ck(MPI_Finalize());}
