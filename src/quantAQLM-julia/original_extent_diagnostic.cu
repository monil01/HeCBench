
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#define main source_quantAQLM_main
#include "/home/imo/HeCBench_julia/src/quantAQLM-cuda/main.cu"
#undef main
int main(){
 const int m=12288,k=4096,b=4,G=16;const size_t nc=m*k/8,ni=b*k,nb=3*65536*8;
 std::vector<uint16_t>input(nc+ni+nb);std::ifstream f("/home/imo/HeCBench_julia/src/quantAQLM-julia/study_case1_u16.bin",std::ios::binary);f.read((char*)input.data(),input.size()*2);if(!f)return 2;
 uint16_t *A,*B,*book,*C;cudaMalloc(&A,nc*2);cudaMalloc(&B,ni*2);cudaMalloc(&book,nb*2);cudaMalloc(&C,(m+G)*2);
 cudaMemcpy(A,input.data(),nc*2,cudaMemcpyHostToDevice);cudaMemcpy(B,input.data()+nc,ni*2,cudaMemcpyHostToDevice);cudaMemcpy(book,input.data()+nc+ni,nb*2,cudaMemcpyHostToDevice);
 std::vector<uint16_t>out(m+G,0x7b00);cudaMemcpy(C,out.data(),out.size()*2,cudaMemcpyHostToDevice);
 int sms;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);int waves=0,tm;do{waves++;tm=ceildiv(m,waves*sms);}while(tm>THREAD_M);int blocks=ceildiv(m,tm),threads=32*tm;
 Code1x16MatVec<<<blocks,threads,16*32*F>>>((int4*)A,(int4*)B,(int4*)C,(int4*)book,m,k,make_int4(4096,8192,12288,122880),65536);
 cudaDeviceSynchronize();auto err=cudaGetLastError();cudaMemcpy(out.data(),C,out.size()*2,cudaMemcpyDeviceToHost);
 int beyond=0,tail=0;for(int i=b*m/32;i<m;i++)beyond+=out[i]!=0x7b00;for(int i=m;i<m+G;i++)tail+=out[i]!=0x7b00;
 std::ofstream raw("/home/imo/HeCBench_julia/.porting-state/portable_julia/diagnostics/quantAQLM/original_extent_u16.bin",std::ios::binary);raw.write((char*)out.data(),out.size()*2);
 printf("{\"logical_original_total_half_capacity\":%d,\"one_kernel_output_rows\":%d,\"writes_beyond_original_total_capacity\":%d,\"allocated_diagnostic_backing_half\":%d,\"backing_tail_guard_changes\":%d,\"SMs\":%d,\"blocks\":%d,\"threads\":%d,\"cuda_status\":%d}\n",b*m/32,m,beyond,m+G,tail,sms,blocks,threads,int(err));
 return err!=cudaSuccess;
}
