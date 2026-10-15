#include <iris/iris.hpp>
#include <memory>
#include "reference.h"
int main(int argc,char**argv) {
 if(argc!=9){printf("Usage: %s <number of MLP layers> <batch size> <input feature size> <hidden dimension> <number of outputs> <use relu> <use bias> <repeat>\n",argv[0]);return 1;}
 int layers=atoi(argv[1]),batch=atoi(argv[2]),features=atoi(argv[3]),hidden=atoi(argv[4]),outputs=atoi(argv[5]),reluf=atoi(argv[6]),biasf=atoi(argv[7]),repeat=atoi(argv[8]);
 if(layers<1||batch<1||features<1||hidden<1||outputs<1||repeat<1)return 1;
 g.seed(123);std::vector<float> input(size_t(batch)*features);
 for(float&x:input)x=distr(g);
 std::vector<int> sizes(layers,hidden);sizes.back()=outputs;
 std::vector<std::vector<float>> w(layers),bias(layers),out(layers);
 for(int l=0;l<layers;++l){w[l].resize(size_t(l?sizes[l-1]:features)*sizes[l]);for(float&x:w[l])x=distr(g);}
 for(int l=0;l<layers;++l){bias[l].resize(sizes[l]);for(float&x:bias[l])x=distr(g);out[l].resize(size_t(batch)*sizes[l]);}
 iris::Platform platform;platform.init(&argc,&argv,1);
 int error=0;
 {
 iris::DMem din(input.data(),input.size()*sizeof(float));
 std::vector<std::unique_ptr<iris::DMem>> dw,db,dout;
 for(int l=0;l<layers;++l) {
  dw.emplace_back(new iris::DMem(w[l].data(),w[l].size()*4)); db.emplace_back(new iris::DMem(bias[l].data(),bias[l].size()*4));dout.emplace_back(new iris::DMem(out[l].data(),out[l].size()*4));
 }
 auto forward=[&](bool flush) {
 for(int l=0;l<layers;++l){
  int ni=l?sizes[l-1]:features,no=sizes[l];
  auto* in=l?dout[l-1].get():&din;
  void*p[]={in,dw[l].get(),db[l].get(),dout[l].get(),&batch,&ni,&no,&biasf,&reluf};
  int pi[]={iris_r,iris_r,iris_r,iris_w,sizeof(batch),sizeof(ni),sizeof(no),sizeof(biasf),sizeof(reluf)};
  size_t gws[]={((out[l].size()+255)/256)*256},lws[]={256};
  iris::Task task;task.kernel("mlp_layer",1,nullptr,gws,lws,9,p,pi);
  if(flush&&l==layers-1)task.flush_out(*dout[l]);
  if(task.submit(iris_gpu,nullptr,1)!=0)return 1;
 }return 0;
 };
 for(int i=0;i<300;++i)error|=forward(i==299);
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i)error|=forward(i==repeat-1);
 double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeat;
 float* ref=reference(layers,batch,features,hidden,outputs,reluf,biasf,repeat);
 double worst=0;
 for(size_t i=0;i<out.back().size();++i){
  double delta=std::abs(double(out.back()[i])-ref[i]);worst=std::max(worst,delta);
  if(!std::isfinite(out.back()[i])||!std::isfinite(ref[i])||delta>1e-2)++error;
  if(std::getenv("IRIS_MLP_DUMP"))printf("VALUE %zu %.9g\n",i,out.back()[i]);
 }
 delete[]ref;
 printf("Whole-output %zu values max_abs %.12g %s\n",out.back().size(),worst,error?"FAIL":"PASS");
 printf("Average execution time of MLP: %lf (us)\n",us);
 }
 platform.finalize();return error?1:0;
}
