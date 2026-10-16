#define main original_hecbench_main
#include "../maxpool3d-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int w=argc>1?atoi(argv[1]):256,h=argc>2?atoi(argv[2]):256,count=argc>3?atoi(argv[3]):16,repeats=argc>4?atoi(argv[4]):10;
 if(w<2||h<2||count<1||w%2||h%2||repeats<1)return 2;int ow=w/2,oh=h/2;size_t ni=(size_t)w*h*count,no=(size_t)ow*oh*count;
 std::vector<float> host(ni),expected(no,0),got(no);for(size_t i=0;i<ni;i++)host[i]=(float)((long long)(37*i)%511-255)/255.f;
 for(int z=0;z<count;z++)for(int y=0;y<oh;y++)for(int x=0;x<ow;x++){float v=0;for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)v=fmaxf(v,host[((z*h+2*y+dy)*w)+2*x+dx]);expected[(z*oh+y)*ow+x]=v;}
 float*a,*b;if(cudaMalloc(&a,ni*sizeof(float))!=cudaSuccess||cudaMalloc(&b,no*sizeof(float))!=cudaSuccess)return 3;cudaMemcpy(a,host.data(),ni*sizeof(float),cudaMemcpyHostToDevice);
 dim3 block(8,8,4),grid((ow+7)/8,(oh+7)/8,(count+3)/4);auto launch=[&](){maxpool3d<<<grid,block>>>(a,b,2,2,2,2,count,w,h,ow,oh);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(b,0,no*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),b,no*sizeof(float),cudaMemcpyDeviceToHost);double maxabs=0;bool ok=status==cudaSuccess;for(size_t i=0;i<no;i++){double e=fabs((double)got[i]-expected[i]);maxabs=fmax(maxabs,e);ok &=std::isfinite(got[i])&&e==0;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),no,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",no,maxabs,repeats,filename,no);
 }cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
