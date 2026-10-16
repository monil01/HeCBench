#define main hecbench_original_main
#include "minmax-cuda/main.cu"
#undef main
#include <vector>
std::vector<float> evaluate_minmax(thrust::device_vector<vec_2d<float>>&device,std::vector<vec_2d<float>>&points,int phase){
 size_t low,high;
 if(phase==0){auto a=thrust::min_element(thrust::device,device.begin(),device.end(),compare<float>());auto b=thrust::max_element(thrust::device,device.begin(),device.end(),compare<float>());low=thrust::distance(device.begin(),a);high=thrust::distance(device.begin(),b);}
 else{auto a=thrust::minmax_element(thrust::device,device.begin(),device.end(),compare<float>());low=thrust::distance(device.begin(),a.first);high=thrust::distance(device.begin(),a.second);}
 return {points[low].x,points[low].y,points[high].x,points[high].y};
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):10;std::vector<vec_2d<float>>points(n);for(int i=0;i<n;i++)points[i]={float((i*37)%n),float((i*19)%n)/float(n)};
 thrust::device_vector<vec_2d<float>>device(points);auto low=std::min_element(points.begin(),points.end(),compare<float>());auto high=std::max_element(points.begin(),points.end(),compare<float>());std::vector<float>expected={low->x,low->y,high->x,high->y};for(int p=0;p<2;p++)evaluate_minmax(device,points,p);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[2];bool correct=true;std::vector<float>values;
  for(int phase=0;phase<2;phase++){
   cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();std::vector<float>result;for(int r=0;r<repeat;r++)result=evaluate_minmax(device,points,phase);cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[phase]=std::chrono::duration<double>(stop-start).count()/repeat;correct &= result==expected;values.insert(values.end(),result.begin(),result.end());
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(float),8,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":8,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_extrema_operations_host_results_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":8,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],correct?"true":"false",repeat,times[0],times[1],filename);if(!correct)return 1;
 }return 0;
}
