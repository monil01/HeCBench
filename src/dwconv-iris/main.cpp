#include <iris/iris.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static bool forward(int m, int n, int c, int h, int w, int k, int repeat) {
  int oc = c * m, oh = h + 2 - k + 1, ow = w + 2 - k + 1;
  int total = n * oc * oh * ow;
  std::vector<float> input(n*c*h*w), weight(oc*k*k), bias(oc), output(total);
  std::srand(123);
  for (float& x : input) x = std::rand() / (float)RAND_MAX;
  for (float& x : weight) x = std::rand() / (float)RAND_MAX;
  for (float& x : bias) x = std::rand() / (float)RAND_MAX;
  iris::DMem mi(input.data(), input.size()*sizeof(float));
  iris::DMem mw(weight.data(), weight.size()*sizeof(float));
  iris::DMem mb(bias.data(), bias.size()*sizeof(float));
  iris::DMem mo(output.data(), output.size()*sizeof(float));
  bool enabled=true;
  int stride=1, pad=1;
  size_t gws[]={size_t((total+255)/256)*256}, lws[]={256};
  auto start=std::chrono::steady_clock::now();
  for (int i=0;i<repeat;++i) {
    void* args[]={&mi,&mo,&mw,&mb,&enabled,&total,&oc,&m,&w,&h,&ow,&oh,&k,&k,&stride,&stride,&pad,&pad};
    int info[]={iris_r,iris_w,iris_r,iris_r,sizeof(enabled),sizeof(total),sizeof(oc),sizeof(m),sizeof(w),sizeof(h),sizeof(ow),sizeof(oh),sizeof(k),sizeof(k),sizeof(stride),sizeof(stride),sizeof(pad),sizeof(pad)};
    iris::Task task;
    task.kernel("conv_depthwise2d_forward_kernel",1,nullptr,gws,lws,18,args,info);
    if(i==repeat-1) task.flush_out(mo);
    if(task.submit(iris_gpu,nullptr,1)!=0) return false;
  }
  auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  float sum=0;
  for(float x:output) { if(!std::isfinite(x)) return false; sum+=x; }
  std::printf("Average execution time of dwconv2d_forward kernel: %f (ms)\n",ns*1e-6/repeat);
  std::printf("Checksum = %f\n",sum/total);
  // Complete scalar reference with a different traversal from the flattened
  // device index. This complements the native checksum comparison.
  double max_error=0;
  for(int batch=0;batch<n;++batch) for(int ch=0;ch<oc;++ch)
    for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) {
      float ref=bias[ch];
      for(int ky=0;ky<k;++ky) for(int kx=0;kx<k;++kx) {
        int iy=y-1+ky,ix=x-1+kx;
        if(iy>=0 && iy<h && ix>=0 && ix<w)
          ref+=weight[ch*k*k+ky*k+kx]*input[((batch*c+ch/m)*h+iy)*w+ix];
      }
      float got=output[((batch*oc+ch)*oh+y)*ow+x];
      max_error=std::fmax(max_error,std::fabs(double(got)-ref));
    }
  std::printf("Reference max absolute error = %.9g\n",max_error);
  return max_error<=1e-5;
}
int main(int argc,char** argv) {
  if(argc!=6) { std::printf("Usage: %s <batch size> <number of input channels> <input height> <input width> <repeat>\n",argv[0]); return 1; }
  int n=std::atoi(argv[1]),c=std::atoi(argv[2]),h=std::atoi(argv[3]),w=std::atoi(argv[4]),repeat=std::atoi(argv[5]);
  if(n<=0 || c<=0 || h<3 || w<3 || repeat<=0) return 1;
  iris::Platform platform; if(platform.init(&argc,&argv,1)!=0) return 1;
  bool ok=true;
  for(int m=1;m<=4;++m) for(int k=1;k<=5;k+=2) {
    std::printf("batch = %d, input channel = %d, height = %d, width = %d, kernel size = %d, output channel = %d\n",n,c,h,w,k,m*c);
    ok=forward(m,n,c,h,w,k,repeat)&&ok;
  }
  platform.finalize(); return ok?0:1;
}
