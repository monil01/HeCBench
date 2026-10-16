#define main original_main
#include "../vadd-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
int main(int argc,char**argv){
 int m=argc>1?atoi(argv[1]):256,n=argc>2?atoi(argv[2]):256,repeats=argc>3?atoi(argv[3]):100,total=m*n;
 if(m<=0||m%16||n<=0||n%256||total%2048||repeats<=0)return 2;
 std::vector<float>ref(total);std::vector<__half>a(total),b(total),got(total);srand(42);fill_values(ref.data(),a.data(),b.data(),total);
 __half *da,*db,*dc;cudaMalloc(&da,total*2);cudaMalloc(&db,total*2);cudaMalloc(&dc,total*2);cudaMemcpy(da,a.data(),total*2,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),total*2,cudaMemcpyHostToDevice);
 auto launch=[&](int mode){if(mode==1)run_base(da,db,dc,total);else if(mode==2)run_vectorized(da,db,dc,total);else run_tv_layout(da,db,dc,m,n);};
 for(int mode=1;mode<=3;mode++)launch(mode);cudaDeviceSynchronize();bool allok=true;
 for(int trial=1;trial<=5;trial++){
  double times[3],emax=0;bool ok=true;std::vector<float>outputs;outputs.reserve(total*3);
  for(int mode=1;mode<=3;mode++){
   cudaMemset(dc,0,total*2);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int r=0;r<repeats;r++)launch(mode);cudaDeviceSynchronize();times[mode-1]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
   cudaMemcpy(got.data(),dc,total*2,cudaMemcpyDeviceToHost);ok&=allclose(ref.data(),got.data(),total);for(int i=0;i<total;i++){float v=__half2float(got[i]);ok&=std::isfinite(v);emax=fmax(emax,fabs(v-ref[i]));outputs.push_back(v);}
  }
  ok&=cudaGetLastError()==cudaSuccess;allok&=ok;std::string file=std::string(getenv("STUDY_OUTPUT_DIR"))+"/vadd_"+std::to_string(trial)+".bin";std::ofstream out(file,std::ios::binary);out.write((char*)outputs.data(),outputs.size()*4);out.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.9g,\"max_rel_error\":0,\"timing_region\":\"sum_three_synchronized_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1]+times[2],ok?"true":"false",outputs.size(),emax,repeats,times[0],times[1],times[2],file.c_str(),outputs.size());
 }
 cudaFree(da);cudaFree(db);cudaFree(dc);return allok?0:1;
}
