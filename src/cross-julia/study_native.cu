// Native cross3 FP32 configuration only; cross1/cross2/FP64 not measured.
#define main original_hecbench_main
#include "../cross-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):1048576,repeats=argc>2?atoi(argv[2]):10;if(n<1||repeats<1)return 2;size_t size=(size_t)3*n;
 std::vector<float>a(size),b(size),expected(size),got(size);for(size_t i=0;i<size;i++){a[i]=((long long)((17*i)%33)-16)/8.f;b[i]=((long long)((13*i)%31)-15)/16.f;}
 for(int r=0;r<n;r++){size_t p=3*(size_t)r;expected[p]=a[p+1]*b[p+2]-a[p+2]*b[p+1];expected[p+1]=a[p+2]*b[p]-a[p]*b[p+2];expected[p+2]=a[p]*b[p+1]-a[p+1]*b[p];}
 float*da,*db,*out;if(cudaMalloc(&da,size*sizeof(float))!=cudaSuccess||cudaMalloc(&db,size*sizeof(float))!=cudaSuccess||cudaMalloc(&out,size*sizeof(float))!=cudaSuccess)return 3;
 cudaMemcpy(da,a.data(),size*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),size*sizeof(float),cudaMemcpyHostToDevice);
 auto launch=[&](){cross3_kernel<float><<<(n+255)/256,256>>>(n,out,da,db);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(out,0,size*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),out,size*sizeof(float),cudaMemcpyDeviceToHost);double maxabs=0,maxrel=0;bool ok=status==cudaSuccess;
  for(size_t i=0;i<size;i++){double e=fabs((double)got[i]-expected[i]);maxabs=fmax(maxabs,e);maxrel=fmax(maxrel,e/fmax(fabs((double)expected[i]),1e-6));ok &=std::isfinite(got[i])&&e<=1e-6;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),size,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",size,maxabs,maxrel,repeats,filename,size);
 }cudaFree(da);cudaFree(db);cudaFree(out);puts(allok?"PASS":"FAIL");return allok?0:1;
}
