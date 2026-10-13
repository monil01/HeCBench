#include <iris/iris.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef NPTS
#define NPTS (2048*8)
#endif
#define NDIM 128
int main(int argc,char** argv) {
  if(argc!=2) { printf("Usage: %s <repeat>\n",argv[0]); return 1; }
  int repeat=atoi(argv[1]); if(repeat<1) return 1;
  std::vector<float> pts1(size_t(NPTS)*NDIM),pts2(pts1.size()),score(NPTS),ref(NPTS);
  std::vector<int> index(NPTS),refidx(NPTS);
  // Generate using the source's initial C RNG state before IRIS reseeds it.
  for(int i=0;i<NPTS;i++) {
    float sum1=0,sum2=0;
    for(int d=0;d<NDIM;d++) {
      sum1 += pts1[i*NDIM+d]=float(rand())/RAND_MAX;
      sum2 += pts2[i*NDIM+d]=float(rand())/RAND_MAX;
    }
    sum1=sqrt(NDIM)/sum1; sum2=sqrt(NDIM)/sum2;
    for(int d=0;d<NDIM;d++) { pts1[i*NDIM+d]*=sum1; pts2[i*NDIM+d]*=sum2; }
  }
  for(int i=0;i<NPTS;i++) for(int j=0;j<NPTS;j++) {
    float s=0; for(int d=0;d<NDIM;d++) s+=pts1[i*NDIM+d]*pts2[j*NDIM+d];
    if(s>ref[i]) { ref[i]=s; refidx[i]=j; }
  }
  iris::Platform platform; if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  bool gpu=false;
  for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0; char name[256]={}; size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes); iris_device_info(d,iris_type,&type,&bytes); iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name); gpu |= backend==3 && type==256;
  }
  if(!gpu) return 1;
  iris::DMem a(pts1.data(),pts1.size()*sizeof(float)),b(pts2.data(),pts2.size()*sizeof(float)),s(score.data(),score.size()*sizeof(float)),idx(index.data(),index.size()*sizeof(int));
  const int widths[]={128,16,16,16,16,16,32,32,32,32};
  const int heights[]={1,16,16,16,16,16,8,8,4,8};
  bool ok=true;
  FILE* artifact=nullptr; if(const char* path=getenv("HECBENCH_ARTIFACT")) { artifact=fopen(path,"wb"); if(!artifact) return 3; }
  for(int k=0;k<10;k++) {
    char name[32]; snprintf(name,sizeof(name),"Match%d",k+1);
    size_t gws[]={NPTS,size_t(heights[k])},lws[]={size_t(widths[k]),size_t(heights[k])};
    auto start=std::chrono::steady_clock::now();
    for(int r=0;r<repeat;r++) {
      iris::Task task; void* args[]={&a,&b,&s,&idx}; int info[]={iris_r,iris_r,iris_w,iris_w};
      if(task.kernel(name,2,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS) return 2;
      if(r==repeat-1) { task.flush_out(s); task.flush_out(idx); }
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }
    printf("MatchGPU%d: %f ms\n",k+1,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat);
    int diff=0; double maxerr=0;
    for(int i=0;i<NPTS;i++) { diff += index[i]!=refidx[i]; maxerr=fmax(maxerr,fabs(score[i]-ref[i])); ok &= std::isfinite(score[i]) && fabs(score[i]-ref[i])<1e-4; }
    printf("Number of incorrect matches: %d\n",diff);
    printf("FULL_CHECK kernel=%d points=%d score_max_abs=%.9g %s\n",k+1,NPTS,maxerr,(!diff && maxerr<1e-4) ? "PASS" : "FAIL");
    ok &= diff==0;
    if(artifact) { fwrite(index.data(),sizeof(int),NPTS,artifact); fwrite(score.data(),sizeof(float),NPTS,artifact); }
  }
  if(artifact) fclose(artifact);
  int errors=platform.error_count(); platform.finalize(); return ok && !errors ? 0 : 1;
}
