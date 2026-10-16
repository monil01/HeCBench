#define main hecbench_original_main
#include "lif-cuda/main.cu"
#undef main
int main(int argc,char**argv){
 int peritem=argc>1?atoi(argv[1]):64,items=argc>2?atoi(argv[2]):64,steps=argc>3?atoi(argv[3]):10,n=peritem*items;
 float dt=.1f,rc=10.f,ref=2.f;std::vector<float>encode(items),bias(peritem),gain(peritem),voltage(n),reftime(n),spikes(n),refv(n),refr(n),refs(n);srand(123);
 for(int i=0;i<items;i++)encode[i]=rand()/float(RAND_MAX);for(int i=0;i<n;i++){voltage[i]=1.f+rand()/float(RAND_MAX);reftime[i]=rand()%5/10.f;}for(int i=0;i<peritem;i++){bias[i]=rand()/float(RAND_MAX);gain[i]=rand()/float(RAND_MAX)+.5f;}refv=voltage;refr=reftime;
 for(int s=0;s<steps;s++)reference(n,peritem,dt,encode.data(),refv.data(),refr.data(),rc,ref,bias.data(),gain.data(),refs.data());
 float *de,*db,*dg,*dv,*dr,*ds;cudaMalloc(&de,items*4);cudaMalloc(&db,peritem*4);cudaMalloc(&dg,peritem*4);cudaMalloc(&dv,n*4);cudaMalloc(&dr,n*4);cudaMalloc(&ds,n*4);cudaMemcpy(de,encode.data(),items*4,cudaMemcpyHostToDevice);cudaMemcpy(db,bias.data(),peritem*4,cudaMemcpyHostToDevice);cudaMemcpy(dg,gain.data(),peritem*4,cudaMemcpyHostToDevice);cudaMemcpy(dv,voltage.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dr,reftime.data(),n*4,cudaMemcpyHostToDevice);
 lif<<<(n+255)/256,256>>>(n,peritem,dt,de,dv,dr,rc,ref,db,dg,ds);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  cudaMemcpy(dv,voltage.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dr,reftime.data(),n*4,cudaMemcpyHostToDevice);cudaMemset(ds,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int s=0;s<steps;s++)lif<<<(n+255)/256,256>>>(n,peritem,dt,de,dv,dr,rc,ref,db,dg,ds);cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();double time=std::chrono::duration<double>(stop-start).count()/steps;
  std::vector<float>values(3*n),expected;cudaMemcpy(values.data(),dv,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(values.data()+n,dr,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(values.data()+2*n,ds,n*4,cudaMemcpyDeviceToHost);expected.insert(expected.end(),refv.begin(),refv.end());expected.insert(expected.end(),refr.begin(),refr.end());expected.insert(expected.end(),refs.begin(),refs.end());bool correct=true;float error=0;for(int i=0;i<3*n;i++){float e=fabsf(values[i]-expected[i]);error=fmaxf(error,e);correct &= std::isfinite(values[i])&&e<=1e-4f+1e-5f*fabsf(expected[i]);}
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(float),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"timing_region\":\"synchronized_neuron_steps_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,correct?"true":"false",3*n,error,steps,filename,3*n);if(!correct)return 1;
 }cudaFree(de);cudaFree(db);cudaFree(dg);cudaFree(dv);cudaFree(dr);cudaFree(ds);return 0;
}
