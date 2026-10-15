#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
struct Tensor{
 std::vector<float>v;std::unique_ptr<iris::DMem>m;
 Tensor(size_t n):v(n,0){}
 void bind(){m.reset(new iris::DMem(v.data(),v.size()*sizeof(float)));}
 void file(const std::string&p,bool out=false){FILE*f=fopen(p.c_str(),out?"wb":"rb");if(!f)throw std::runtime_error(p);size_t n=out?fwrite(v.data(),4,v.size(),f):fread(v.data(),4,v.size(),f);fclose(f);if(n!=v.size())throw std::runtime_error("tensor size "+p);}
};
void launch(const char*name,size_t n,std::vector<void*>args,std::vector<int>info,std::vector<Tensor*>outputs={}){
 size_t g[]={((n+255)/256)*256},l[]={256};iris::Task t;t.kernel(name,1,nullptr,g,l,args.size(),args.data(),info.data());for(auto*out:outputs)t.flush_out(*out->m);t.submit(iris_gpu,nullptr,1);
}
int main(int argc,char**argv){
 if(argc!=8){fprintf(stderr,"usage: main B C O H W groups tensor_prefix\n");return 2;}
 int B=atoi(argv[1]),C=atoi(argv[2]),O=atoi(argv[3]),H=atoi(argv[4]),W=atoi(argv[5]),G=atoi(argv[6]),P=G*27;std::string p=argv[7];
 if(B<=0||C<=0||O<=0||H<=0||W<=0||G<=0||C%G)return 2;
 iris::Platform platform;platform.init(&argc,&argv,1);
 try{
 Tensor x(size_t(B)*C*H*W),w(size_t(O)*C*9),bias(O),pw(size_t(P)*C*9),pb(P),pred(size_t(B)*P*H*W),y(size_t(B)*O*H*W),target(y.v.size()),go(y.v.size()),mean(1),gx(x.v.size()),gw(w.v.size()),gb(O),gp(pred.v.size()),gpw(pw.v.size()),gpb(P);
 for(auto pair:std::vector<std::pair<Tensor*,const char*>>{{&x,"x"},{&w,"w"},{&bias,"bias"},{&pw,"pw"},{&pb,"pb"},{&target,"target"}})pair.first->file(p+"."+pair.second);
 for(auto*t:{&x,&w,&bias,&pw,&pb,&pred,&y,&target,&go,&mean,&gx,&gw,&gb,&gp,&gpw,&gpb})t->bind();
 launch("conv",pred.v.size(),{x.m.get(),pw.m.get(),pb.m.get(),pred.m.get(),&B,&C,&P,&H,&W},{iris_r,iris_r,iris_r,iris_w,sizeof(B),sizeof(C),sizeof(P),sizeof(H),sizeof(W)});
 launch("deform",y.v.size(),{x.m.get(),w.m.get(),bias.m.get(),pred.m.get(),y.m.get(),&B,&C,&O,&H,&W,&G},{iris_r,iris_r,iris_r,iris_r,iris_w,sizeof(B),sizeof(C),sizeof(O),sizeof(H),sizeof(W),sizeof(G)},{&y});
 int n=y.v.size();launch("loss",1,{y.m.get(),target.m.get(),go.m.get(),mean.m.get(),&n},{iris_r,iris_r,iris_w,iris_w,sizeof(n)},{&mean});
 launch("deform_backward",y.v.size(),{x.m.get(),w.m.get(),pred.m.get(),go.m.get(),gx.m.get(),gw.m.get(),gb.m.get(),gp.m.get(),&B,&C,&O,&H,&W,&G},{iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(B),sizeof(C),sizeof(O),sizeof(H),sizeof(W),sizeof(G)},{&gw,&gb,&gp});
 launch("conv_backward",pred.v.size(),{x.m.get(),pw.m.get(),gp.m.get(),gx.m.get(),gpw.m.get(),gpb.m.get(),&B,&C,&P,&H,&W},{iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,sizeof(B),sizeof(C),sizeof(P),sizeof(H),sizeof(W)},{&gx,&gpw,&gpb});
 for(auto pair:std::vector<std::pair<Tensor*,const char*>>{{&y,"out"},{&mean,"mean"},{&gx,"gx"},{&gw,"gw"},{&gb,"gb"},{&gp,"gp"},{&gpw,"gpw"},{&gpb,"gpb"}})pair.first->file(p+"."+pair.second,true);
 printf("IRIS HIP DCN forward, loss and complete backward computed\n");
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());platform.finalize();return 1;}platform.finalize();return 0;
}
