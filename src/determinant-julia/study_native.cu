#define main original_hecbench_main
#include "../determinant-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=11,repeats=10;float host[121]={
     2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
    -2.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    -2.,  0.,  6.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    -2.,  0.,  2.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
    -2.,  0.,  2.,  4.,  10., 10., 10., 10., 10., 10., 10.,
    -2.,  0.,  2.,  4.,  10., 12., 12., 12., 12., 12., 12.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 14., 14., 14., 14.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 16., 16., 16.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 18., 18.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 20.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 22.
  };std::vector<float>got(68),expected(68);std::vector<double>L(121,0);int ix=0;
for(int col=0;col<n;col++)for(int row=col;row<n;row++){double v=host[col*n+row];for(int k=0;k<col;k++)v-=L[k*n+row]*L[k*n+col];L[col*n+row]=row==col?sqrt(v):v/L[col*n+col];}
double rootref=1;for(int col=0;col<n;col++){rootref*=L[col*n+col];for(int row=col;row<n;row++)expected[ix++]=L[col*n+row];}expected[66]=rootref;expected[67]=rootref*rootref;
float *a,*workspace;int *info,ws=0;cusolverDnHandle_t handle;cusolverDnCreate(&handle);cudaMalloc(&a,121*4);cudaMalloc(&info,4);cusolverDnSpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,a,n,&ws);cudaMalloc(&workspace,ws*4);float root=0;cusolverStatus_t solver_status=CUSOLVER_STATUS_SUCCESS;

 auto launch=[&](){cudaMemcpy(a,host,121*4,cudaMemcpyHostToDevice);solver_status=cusolverDnSpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,a,n,workspace,ws,info);thrust::device_ptr<float>dev_ptr=thrust::device_pointer_cast(a);typedef thrust::device_vector<float>::iterator Iterator;strided_range<Iterator>pos(dev_ptr,dev_ptr+121,12);root=thrust::reduce(pos.begin(),pos.end(),1.f,thrust::multiplies<float>());};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 float factor[121];int hinfo=-1;cudaMemcpy(factor,a,121*4,cudaMemcpyDeviceToHost);cudaMemcpy(&hinfo,info,4,cudaMemcpyDeviceToHost);if(hinfo!=0||solver_status!=CUSOLVER_STATUS_SUCCESS)return 8;int idx=0;for(int col=0;col<n;col++)for(int row=col;row<n;row++)got[idx++]=factor[col*n+row];got[66]=root;got[67]=root*root;if(!(std::isfinite(got[67])&&fabs(got[67]-2048.f)<.05))return 9;
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.05));ok&=std::isfinite((double)got[j])&&e<=(j<66?1e-5:.05);}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"h2d_factorization_device_diag_product_scalar_d2h_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
