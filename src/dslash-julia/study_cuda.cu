#define PRECISION 1
#define LDIM 6
#include "../dslash-cuda/kernels.cu"
#include <fstream>
#include <iomanip>
#include <string>
size_t sites_on_node,even_sites_on_node,warmups=1;unsigned int verbose=0;std::vector<int>squaresize(4,6);
#define CK(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(err));exit(3);}}while(0)
const double G=-777;
std::vector<double> readfile(std::string f){std::ifstream s(f,std::ios::binary|std::ios::ate);size_t n=s.tellg();s.seekg(0);std::vector<double>v(n/8);s.read((char*)v.data(),n);return v;}
template<class T>struct Buffer{T*base;int n,g;std::vector<T>h;Buffer(const std::vector<double>&v,int gg,double guard):n(v.size()),g(gg),h(n+2*g,T(guard)){for(int i=0;i<n;i++)h[g+i]=T(v[i]);CK(cudaMalloc(&base,h.size()*sizeof(T)));CK(cudaMemcpy(base,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}T*data(){return base+g;}void fetch(){CK(cudaMemcpy(h.data(),base,h.size()*sizeof(T),cudaMemcpyDeviceToHost));}void reset(){std::fill(h.begin(),h.end(),T(G));CK(cudaMemcpy(base,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}};
void append(std::vector<double>&out,const std::vector<float>&v){for(auto x:v)out.push_back(x);}void append(std::vector<double>&out,const std::vector<size_t>&v){for(auto x:v)out.push_back(x);}
int main(){std::string root=std::string(__FILE__).substr(0,std::string(__FILE__).find_last_of('/'));auto fixture=readfile(root+"/fixture.bin"),oracle=readfile(root+"/oracle.bin");size_t pos=0;
 struct Case{int L,n;Buffer<float>*s,*f,*l,*dst,*fb,*lb;Buffer<size_t>*fw,*b,*fw3,*b3;};std::vector<Case> cases;
 auto take=[&](int n){std::vector<double>v(fixture.begin()+pos,fixture.begin()+pos+n);pos+=n;return v;};
 for(int ci=0;ci<3;ci++){int L=fixture[pos++],n=fixture[pos++];Case c;c.L=L;c.n=n;c.s=new Buffer<float>(take(n*6),6,G);c.f=new Buffer<float>(take(n*72),18,G);c.l=new Buffer<float>(take(n*72),18,G);c.fw=new Buffer<size_t>(take(n*4),4,999999);c.b=new Buffer<size_t>(take(n*4),4,999999);c.fw3=new Buffer<size_t>(take(n*4),4,999999);c.b3=new Buffer<size_t>(take(n*4),4,999999);
 sites_on_node=n;even_sites_on_node=n/2;nx=ny=nz=nt=L;squaresize.assign(4,L);std::vector<size_t>a(n*4),b(n*4),d(n*4),e(n*4);set_neighbors(a.data(),b.data(),d.data(),e.data());if(!std::equal(a.begin(),a.end(),c.fw->h.begin()+4)||!std::equal(b.begin(),b.end(),c.b->h.begin()+4)||!std::equal(d.begin(),d.end(),c.fw3->h.begin()+4)||!std::equal(e.begin(),e.end(),c.b3->h.begin()+4))return 4;
 c.dst=new Buffer<float>(std::vector<double>(n*6,G),6,G);c.fb=new Buffer<float>(std::vector<double>(n*72,G),18,G);c.lb=new Buffer<float>(std::vector<double>(n*72,G),18,G);cases.push_back(c);}
 auto back=[](Case&c){make_back<<<c.n/2,1>>>((su3_matrix*)c.f->data(),(su3_matrix*)c.l->data(),c.b->data(),c.b3->data(),(su3_matrix*)c.fb->data(),(su3_matrix*)c.lb->data(),c.n/2);CK(cudaDeviceSynchronize());};
 auto hop=[](Case&c){dslash<<<(c.n/2+255)/256,256>>>((su3_matrix*)c.f->data(),(su3_matrix*)c.l->data(),(su3_matrix*)c.fb->data(),(su3_matrix*)c.lb->data(),(su3_vector*)c.s->data(),(su3_vector*)c.dst->data(),c.fw->data(),c.b->data(),c.fw3->data(),c.b3->data(),c.n/2);CK(cudaDeviceSynchronize());};for(auto&c:cases){back(c);hop(c);}
 for(int trial=1;trial<=5;trial++){std::vector<double>prefix,got,path;double elapsed=0;for(auto&c:cases){c.dst->reset();c.fb->reset();c.lb->reset();back(c);auto start=std::chrono::steady_clock::now();for(int r=0;r<100;r++)hop(c);double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/100;elapsed+=dt;path.push_back(dt);
 prefix.push_back(c.L);prefix.push_back(c.n);for(auto v:{c.s,c.f,c.l}){v->fetch();for(int i=0;i<v->n;i++)prefix.push_back(v->h[v->g+i]);}for(auto v:{c.fw,c.b,c.fw3,c.b3}){v->fetch();for(int i=0;i<v->n;i++)prefix.push_back(v->h[v->g+i]);}
 for(auto v:{c.dst,c.fb,c.lb}){v->fetch();append(got,v->h);}for(auto v:{c.s,c.f,c.l})append(got,v->h);for(auto v:{c.fw,c.b,c.fw3,c.b3})append(got,v->h);}
 bool ok=prefix==fixture&&got.size()==oracle.size();double ma=0,mr=0;for(size_t i=0;i<got.size();i++){double er=std::abs(got[i]-oracle[i]);ma=std::max(ma,er);mr=std::max(mr,er/std::max(std::abs(oracle[i]),1e-30));ok &= std::isfinite(got[i])&&er<=2e-5+2e-6*std::abs(oracle[i]);if(std::abs(oracle[i])>=700)ok &= got[i]==oracle[i];}
 std::vector<double>out=prefix;out.insert(out.end(),got.begin(),got.end());std::string file="output_trial"+std::to_string(trial)+".bin";std::ofstream os(file,std::ios::binary);os.write((char*)out.data(),out.size()*8);os.close();std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"timing_region\":\"host_inclusive_sum3_dslash_100sync_launch_batches_div100_backward_setup_excluded\",\"jit_excluded\":true,\"internal_repeats\":100,\"path_times_seconds\":["<<path[0]<<","<<path[1]<<","<<path[2]<<"],\"output_artifact\":{\"path\":\""<<file<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}\n";
 }
}
