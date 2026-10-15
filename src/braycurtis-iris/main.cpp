#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
template<class T> void io(const std::string& path,std::vector<T>& v,bool write=false) {
  FILE*f=fopen(path.c_str(),write?"wb":"rb");
  if(!f) throw std::runtime_error("cannot open "+path);
  size_t n=write?fwrite(v.data(),sizeof(T),v.size(),f):fread(v.data(),sizeof(T),v.size(),f);
  int extra=write?EOF:fgetc(f); fclose(f);
  if(n!=v.size() || extra!=EOF) throw std::runtime_error("wrong tensor length "+path);
}
template<class T> void run(int B,int M,int N,int D,const std::string&p,bool grad) {
  std::vector<T>x(size_t(B)*M*D),y(size_t(B)*N*D),o(size_t(B)*M*N),sd(o.size()),ss(o.size()),g(o.size()),gx(x.size()),gy(y.size());
  io(p+".x",x);io(p+".y",y);if(grad)io(p+".g",g);
  iris::DMem mx(x.data(),x.size()*sizeof(T)),my(y.data(),y.size()*sizeof(T)),mo(o.data(),o.size()*sizeof(T)),md(sd.data(),sd.size()*sizeof(T)),ms(ss.data(),ss.size()*sizeof(T));
  size_t l[]={256},n[]={((o.size()+255)/256)*256};
  void* a[]={&mx,&my,&mo,&md,&ms,&B,&M,&N,&D};int ai[]={iris_r,iris_r,iris_w,iris_w,iris_w,sizeof(B),sizeof(M),sizeof(N),sizeof(D)};
  iris::Task f;f.kernel(sizeof(T)==4?"forward_f":"forward_d",1,nullptr,n,l,9,a,ai);f.flush_out(mo);f.submit(iris_gpu,nullptr,1);io(p+".out",o,true);
  if(grad){
    iris::DMem mg(g.data(),g.size()*sizeof(T)),mgx(gx.data(),gx.size()*sizeof(T)),mgy(gy.data(),gy.size()*sizeof(T));
    for(int w=0;w<2;++w){
      auto &dest=w?mgy:mgx;size_t nn[]={(((w?gy.size():gx.size())+255)/256)*256};
      void*b[]={&mx,&my,&mg,&md,&ms,&dest,&B,&M,&N,&D,&w};int bi[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,sizeof(B),sizeof(M),sizeof(N),sizeof(D),sizeof(w)};
      iris::Task t;t.kernel(sizeof(T)==4?"backward_f":"backward_d",1,nullptr,nn,l,11,b,bi);t.flush_out(dest);t.submit(iris_gpu,nullptr,1);
    }
    io(p+".gx",gx,true);io(p+".gy",gy,true);
  }
}
int main(int argc,char**argv){
  if(argc!=8){fprintf(stderr,"usage: main float|double B M N D tensor_prefix gradient(0|1)\n");return 2;}
  int B=atoi(argv[2]),M=atoi(argv[3]),N=atoi(argv[4]),D=atoi(argv[5]),g=atoi(argv[7]);
  if(B<=0||M<=0||N<=0||D<=0)return 2;
  std::string dtype=argv[1],prefix=argv[6];
  iris::Platform platform;platform.init(&argc,&argv,1);
  try {if(dtype=="float")run<float>(B,M,N,D,prefix,g);else if(dtype=="double")run<double>(B,M,N,D,prefix,g);else throw std::runtime_error("invalid dtype");}
  catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());platform.finalize();return 1;}
  platform.finalize();return 0;
}
