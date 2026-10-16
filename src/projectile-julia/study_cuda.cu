// Named deterministic fixture; unchanged native CalculateRange device kernel.
#define main hecbench_original_main
#include "../projectile-cuda/Projectile.cu"
#undef main
#include <fstream>
#include <string>
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):1048576,repeats=argc>2?atoi(argv[2]):10;
 if(n<=0||n>num_elements||n%256||repeats<=0)return 2;
 std::vector<Projectile> input(n),output(n),expected(n);
 for(int i=0;i<n;i++)input[i]=Projectile(float(10+i%90),float(10+i%400),1,1,1);
 reference(input.data(),expected.data(),n);
 Projectile *a,*b;cudaMalloc(&a,n*sizeof(Projectile));cudaMalloc(&b,n*sizeof(Projectile));
 cudaMemcpy(a,input.data(),n*sizeof(Projectile),cudaMemcpyHostToDevice);
 CalculateRange<<<n/256,256>>>(a,b);cudaDeviceSynchronize();
 bool allok=true;std::vector<float> artifact(3*n);
 for(int trial=1;trial<=5;trial++){
  cudaMemset(b,0,n*sizeof(Projectile));cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  for(int j=0;j<repeats;j++)CalculateRange<<<n/256,256>>>(a,b);
  cudaError_t status=cudaDeviceSynchronize();
  double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(output.data(),b,n*sizeof(Projectile),cudaMemcpyDeviceToHost);
  bool ok=status==cudaSuccess;double maxabs=0,maxrel=0;
  for(int i=0;i<n;i++){
   ok &= !(output[i]!=expected[i]);
   float vals[3]={output[i].getRange(),output[i].gettotalTime(),output[i].getmaxHeight()};
   float refs[3]={expected[i].getRange(),expected[i].gettotalTime(),expected[i].getmaxHeight()};
   for(int k=0;k<3;k++){artifact[k*n+i]=vals[k];double e=fabs(double(vals[k])-refs[k]);maxabs=fmax(maxabs,e);maxrel=fmax(maxrel,e/fmax(fabs(refs[k]),1.1920929e-7));ok &= std::isfinite(vals[k]);}
  }
  allok &= ok;const char* od=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(od?od:".")+"/projectile_"+std::to_string(trial)+".bin";
  std::ofstream file(path,std::ios::binary);file.write((char*)artifact.data(),artifact.size()*sizeof(float));file.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",3*n,maxabs,maxrel,repeats,path.c_str(),3*n);
 }
 cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
