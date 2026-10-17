#define main original_hecbench_main
#include "../bezier-surface-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=512,repeats=3;std::vector<XYZ> control={{-0.508359f,-0.280288f,-0.160131f},{-0.199345f,-0.515802f,-0.0100263f},{0.133988f,-0.515802f,-0.0100263f},{0.418629f,-0.382526f,-0.407321f},{-0.532679f,-0.182468f,-0.0100263f},{0.0505979f,-0.195094f,1.08832f},{0.383931f,-0.195094f,1.08832f},{0.467321f,-0.182468f,-0.0100263f},{-0.532679f,0.150865f,-0.0100263f},{0.0505979f,0.13824f,1.08832f},{0.383931f,0.13824f,1.08832f},{0.467321f,0.150865f,-0.0100263f},{-0.483195f,0.349859f,-0.0568787f},{-0.199345f,0.484198f,-0.0100263f},{0.133988f,0.484198f,-0.0100263f},{0.30311f,0.389596f,-0.0811675f}};std::vector<XYZ> cpu(n*n);BezierCPU(control.data(),cpu.data(),3,3,n,n);
std::vector<float> expected(3*n*n),got(3*n*n);const double coeff[4]={1,3,3,1};
for(int i=0;i<n;i++)for(int j=0;j<n;j++){double u=(double)i/(n-1),v=(double)j/(n-1),x=0,y=0,z=0;for(int ki=0;ki<=3;ki++)for(int kj=0;kj<=3;kj++){double w=coeff[ki]*pow(u,ki)*pow(1-u,3-ki)*coeff[kj]*pow(v,kj)*pow(1-v,3-kj);XYZ a=control[ki*4+kj];x+=(double)a.x*w;y+=(double)a.y*w;z+=(double)a.z*w;}int k=3*(i*n+j);expected[k]=x;expected[k+1]=y;expected[k+2]=z;}
XYZ *input,*output;cudaMalloc(&input,16*sizeof(XYZ));cudaMalloc(&output,n*n*sizeof(XYZ));cudaMemcpy(input,control.data(),16*sizeof(XYZ),cudaMemcpyHostToDevice);

 auto launch=[&](){BezierGPU<<<2,256>>>(input,output,3,3,n,n);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(output,0,n*n*sizeof(XYZ));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),output,n*n*sizeof(XYZ),cudaMemcpyDeviceToHost);double delta=0,refnorm=0;const float*cpuref=(const float*)cpu.data();for(size_t i=0;i<got.size();i++){delta+=fabs((double)got[i]-cpuref[i]);refnorm+=fabs((double)cpuref[i]);}double l1=delta/refnorm;if(!std::isfinite(l1)||l1>=1e-6)return 6;printf("STUDY_ACCURACY relative_l1_source_cpu=%.12g\n",l1);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-05));ok&=std::isfinite((double)got[j])&&e<=1e-05;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
