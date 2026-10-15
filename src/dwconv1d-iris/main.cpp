#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
struct Tensor{std::vector<float>v;std::unique_ptr<iris::DMem>m;Tensor(size_t n):v(n){}void bind(){m.reset(new iris::DMem(v.data(),v.size()*4));}void file(std::string p,bool out=false){FILE*f=fopen(p.c_str(),out?"wb":"rb");if(!f)throw std::runtime_error(p);size_t n=out?fwrite(v.data(),4,v.size(),f):fread(v.data(),4,v.size(),f);fclose(f);if(n!=v.size())throw std::runtime_error("tensor length");}};
void launch(const char*name,int dim,size_t*g,size_t*l,std::vector<void*>a,std::vector<int>i,std::vector<Tensor*>out={}){iris::Task t;t.kernel(name,dim,nullptr,g,l,a.size(),a.data(),i.data());for(auto*x:out)t.flush_out(*x->m);t.submit(iris_gpu,nullptr,1);}
int main(int argc,char**argv){if(argc!=6)return 2;int B=atoi(argv[1]),C=atoi(argv[2]),T=atoi(argv[3]);float eps=strtof(argv[4],nullptr);std::string p=argv[5];if(B<=0||C<=0||T<4||T>768||B%8||T%4)return 2;iris::Platform platform;platform.init(&argc,&argv,1);
 try{Tensor w(C*T),k(size_t(B)*C*T),x(k.v.size()),go(k.v.size()),values(k.v.size()),gw(k.v.size()),gk(k.v.size()),sumgw(w.v.size()),loss(1);w.file(p+".w");k.file(p+".k");for(auto*t:{&w,&k,&x,&go,&values,&gw,&gk,&sumgw,&loss})t->bind();
 size_t g[]={size_t(T/4),size_t(B*C/8)},l[]={size_t(T/4),1};launch("forward",2,g,l,{w.m.get(),k.m.get(),x.m.get(),&eps,&B,&C,&T},{iris_r,iris_r,iris_w,sizeof(eps),sizeof(B),sizeof(C),sizeof(T)},{&x});
 int n=x.v.size();size_t gl[]={((x.v.size()+255)/256)*256},ll[]={256};launch("loss_grad",1,gl,ll,{x.m.get(),go.m.get(),values.m.get(),&n},{iris_r,iris_w,iris_w,sizeof(n)},{&go});
 g[1]=B*C/2;launch("backward",2,g,l,{w.m.get(),k.m.get(),go.m.get(),gw.m.get(),gk.m.get(),&B,&C,&T},{iris_r,iris_r,iris_r,iris_w,iris_w,sizeof(B),sizeof(C),sizeof(T)},{&gw,&gk});
 size_t gr[]={((w.v.size()+255)/256)*256};launch("reduce_gw",1,gr,ll,{gw.m.get(),sumgw.m.get(),&B,&C,&T},{iris_r,iris_w,sizeof(B),sizeof(C),sizeof(T)},{&sumgw});size_t one[]={1};launch("reduce_loss",1,one,one,{values.m.get(),loss.m.get(),&n},{iris_r,iris_w,sizeof(n)},{&loss});
 for(auto pair:std::vector<std::pair<Tensor*,const char*>>{{&x,"out"},{&go,"go"},{&gw,"batchgw"},{&gk,"gk"},{&sumgw,"gw"},{&loss,"loss"}})pair.first->file(p+"."+pair.second,true);
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());platform.finalize();return 1;}platform.finalize();return 0;}
