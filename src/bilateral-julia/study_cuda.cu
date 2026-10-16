// Original template kernels and CPU references for all three radius values.
#define main hecbench_original_main
#include "../bilateral-cuda/main.cu"
#undef main
#include <vector>
#include <algorithm>
void launch(int r,float*in,float*out,int W,int H,float a,float vi,float vs){dim3 t(16,16),b((W+15)/16,(H+15)/16);if(r==3)bilateralFilter<3><<<b,t>>>(in,out,W,H,a,vi,vs);if(r==6)bilateralFilter<6><<<b,t>>>(in,out,W,H,a,vi,vs);if(r==9)bilateralFilter<9><<<b,t>>>(in,out,W,H,a,vi,vs);}
int main(int argc,char**argv){int W=argc>1?atoi(argv[1]):256,H=argc>2?atoi(argv[2]):256,repeats=argc>5?atoi(argv[5]):5;float vi=argc>3?atof(argv[3]):100.f,vs=argc>4?atof(argv[4]):25.f;if(W<=18||H<=18||vi<=0||vs<=0||repeats<=0)return 2;int n=W*H;float a=.5f/(vi*float(M_PI));std::vector<float>input(n),got(n),allgot;std::vector<std::vector<float>>expected(3,std::vector<float>(n));srand(123);for(auto&x:input)x=rand()%256;
 reference<3>(input.data(),expected[0].data(),W,H,a,vi,vs);reference<6>(input.data(),expected[1].data(),W,H,a,vi,vs);reference<9>(input.data(),expected[2].data(),W,H,a,vi,vs);
 float*din,*dout;cudaMalloc(&din,n*sizeof(float));cudaMalloc(&dout,n*sizeof(float));cudaMemcpy(din,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);for(int r:{3,6,9})launch(r,din,dout,W,H,a,vi,vs);cudaDeviceSynchronize();bool allok=true;int radii[3]={3,6,9};
 for(int t=1;t<=5;t++){allgot.clear();double total=0,ma=0,mr=0,stage[3];bool ok=true;
  for(int c=0;c<3;c++){cudaMemset(dout,0,n*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int k=0;k<repeats;k++)launch(radii[c],din,dout,W,H,a,vi,vs);auto status=cudaDeviceSynchronize();stage[c]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;total+=stage[c];cudaMemcpy(got.data(),dout,n*sizeof(float),cudaMemcpyDeviceToHost);ok &=status==cudaSuccess;for(int i=0;i<n;i++){double e=fabs(double(got[i])-expected[c][i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(double(expected[c][i])),double(1.1920929e-7f)));ok &=std::isfinite(got[i])&&e<=1.e-3;}allgot.insert(allgot.end(),got.begin(),got.end());}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"bilateral_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(allgot.data(),sizeof(float),allgot.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_radius3_6_9_kernel_launch_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"radius_times_seconds\":[%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,total,ok?"true":"false",allgot.size(),ma,mr,repeats,stage[0],stage[1],stage[2],file,allgot.size());
 }
 cudaFree(din);cudaFree(dout);puts(allok?"PASS":"FAIL");return allok?0:1;}
