#define main original_hecbench_main
#include "../coordinates-cuda/main.cu"
#undef main
#include <vector>
template<typename T>int study(int n,int repeats){
 thrust::host_vector<lonlat_2d<T>> host(n);thrust::host_vector<cartesian_2d<T>> got(n),expected(n);
 lonlat_2d<T> origin;origin.x=T(90);origin.y=T(45);
 for(int i=0;i<n;i++){host[i].x=T((17LL*i)%360-180);host[i].y=T((13LL*i)%180-90);expected[i]=to_cartesian_functor<T>(origin)(host[i]);}
 thrust::device_vector<lonlat_2d<T>> input=host;thrust::device_vector<cartesian_2d<T>> output(n);
 auto launch=[&](){thrust::transform(input.cbegin(),input.cend(),output.begin(),to_cartesian_functor<T>(origin));};
 launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 3;bool allok=true;
 for(int trial=1;trial<=5;trial++){
  cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();
  double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;got=output;
  std::vector<T> flat(2*n);double maxabs=0,maxrel=0;bool ok=status==cudaSuccess;
  for(int i=0;i<n;i++){
   flat[i]=got[i].x;flat[n+i]=got[i].y;
   double ex=fabs((double)got[i].x-expected[i].x),ey=fabs((double)got[i].y-expected[i].y);
   maxabs=fmax(maxabs,fmax(ex,ey));ok &=std::isfinite(got[i].x)&&std::isfinite(got[i].y)&&ex<1e-3&&ey<1e-3;
   maxrel=fmax(maxrel,fmax(ex/fmax(fabs((double)expected[i].x),1e-3),ey/fmax(fabs((double)expected[i].y),1e-3)));
  }
  allok &=ok;char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 4;fwrite(flat.data(),sizeof(T),flat.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"%s\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",2*n,maxabs,maxrel,repeats,filename,sizeof(T)==4?"float32":"float64",2*n);
 }
 puts(allok?"PASS":"FAIL");return allok?0:1;
}
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):1048576,r=argc>2?atoi(argv[2]):10;if(n<1||r<1)return 2;return argc>3&&std::string(argv[3])=="float32"?study<float>(n,r):study<double>(n,r);}
