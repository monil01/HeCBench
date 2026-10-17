#define main original_hecbench_main
#include "../compute-score-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int docs=64,vocab=16384,repeats=10,lines=160;std::vector<uint>d1(lines*64,0xffffffff),d2=d1,bloom(16384,0),hhi(lines),hlo(lines);std::vector<ulong>w(vocab,0),info(docs),hs(docs);std::vector<long long>got(docs+2*lines),expected(docs+2*lines,0);
for(int t=0;t<vocab;t++)if(t%5==0){w[t]=((ulong)(t%17)<<24)+(t%31+1);for(int h:{t>>5,t&0x7ffff})bloom[h>>5]|=1u<<(h&31);}
int startline=0;for(int d=0;d<docs;d++){int count=1+d%4,len=128*count-7-d%17;info[d]=((ulong)startline<<32)|(startline+count-1);for(int i=0;i<(len+1)/2;i++){uint t=(17*i+13*d)%vocab,f=1+(11*i+d)%254;d1[startline*64+i]=(t<<8)|f;}for(int i=0;i<len/2;i++){uint t=(29*i+19*d+7)%vocab,f=1+(7*i+d)%254;d2[startline*64+i]=(t<<8)|f;}for(int line=startline;line<startline+count;line++){ulong total=0;for(int j=0;j<64;j++){uint e=d1[line*64+j];if(e!=0xffffffff)total+=w[e>>8]*(ulong)(e&255);e=d2[line*64+j];if(e!=0xffffffff)total+=w[e>>8]*(ulong)(e&255);}expected[d]+=total;expected[docs+line]=total>>32;expected[docs+lines+line]=total&0xffffffff;}startline+=count;}
uint *a,*z,*bh,*hi,*lo;ulong *w1,*w2,*di,*out;
cudaMalloc(&a,d1.size()*4);cudaMalloc(&z,d2.size()*4);cudaMalloc(&bh,bloom.size()*4);cudaMalloc(&hi,lines*4);cudaMalloc(&lo,lines*4);cudaMalloc(&w1,vocab*8);cudaMalloc(&w2,vocab*8);cudaMalloc(&di,docs*8);cudaMalloc(&out,docs*8);
cudaMemcpy(a,d1.data(),d1.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(z,d2.data(),d2.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(bh,bloom.data(),bloom.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(w1,w.data(),vocab*8,cudaMemcpyHostToDevice);cudaMemcpy(w2,w.data(),vocab*8,cudaMemcpyHostToDevice);cudaMemcpy(di,info.data(),docs*8,cudaMemcpyHostToDevice);

 auto launch=[&](){compute<<<lines,8>>>(a,z,w1,w2,bh,hi,lo);reduction<<<1,64>>>(di,hi,lo,out);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,docs*8);cudaMemset(hi,0,lines*4);cudaMemset(lo,0,lines*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(hs.data(),out,docs*8,cudaMemcpyDeviceToHost);cudaMemcpy(hhi.data(),hi,lines*4,cudaMemcpyDeviceToHost);cudaMemcpy(hlo.data(),lo,lines*4,cudaMemcpyDeviceToHost);for(int i=0;i<docs;i++)got[i]=hs[i];for(int i=0;i<lines;i++){got[docs+i]=hhi[i];got[docs+lines+i]=hlo[i];}
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(long long),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
