#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "../memtest-cuda/kernels.h"
int main(){
 const size_t bytes=2*1024*1024,words=bytes/4;char*buf;unsigned*cnt;unsigned long *addr,*ex,*cur,*sec;
 cudaMalloc(&buf,bytes);cudaMalloc(&cnt,4);cudaMalloc(&addr,80);cudaMalloc(&ex,80);cudaMalloc(&cur,80);cudaMalloc(&sec,80);cudaMemset(cnt,0,4);
 auto pipeline=[&](){kernel5_init<<<65536,64>>>(buf,bytes);kernel5_move<<<65536,64>>>(buf,bytes);kernel5_check<<<65536,64>>>(buf,bytes,cnt,addr,ex,cur,sec);};
 pipeline();cudaDeviceSynchronize();unsigned pair[2];cudaMemcpy(pair,buf,8,cudaMemcpyDeviceToHost);unsigned bad=pair[0]^1;cudaMemcpy(buf,&bad,4,cudaMemcpyHostToDevice);cudaMemset(cnt,0,4);kernel5_check<<<65536,64>>>(buf,bytes,cnt,addr,ex,cur,sec);cudaDeviceSynchronize();unsigned err;unsigned long a,e,c,s;cudaMemcpy(&err,cnt,4,cudaMemcpyDeviceToHost);cudaMemcpy(&a,addr,8,cudaMemcpyDeviceToHost);cudaMemcpy(&e,ex,8,cudaMemcpyDeviceToHost);cudaMemcpy(&c,cur,8,cudaMemcpyDeviceToHost);cudaMemcpy(&s,sec,8,cudaMemcpyDeviceToHost);
 bool inject=err==1&&a==(unsigned long)buf&&e==pair[1]&&c==bad&&s==bad;if(!inject)return 7;
 std::vector<unsigned> expected(words),got(words);size_t half=BLOCKSIZE/4/2;
 for(size_t i=0;i<words;i++){size_t block=i/(BLOCKSIZE/4),pos=i%(BLOCKSIZE/4),src=pos<half?(pos+half-8)%half:pos-half,absolute=block*(BLOCKSIZE/4)+src;unsigned p=1u<<((absolute/16)%32);expected[i]=absolute%4<2?p:~p;}
 for(int trial=1;trial<=5;trial++){cudaMemset(cnt,0,4);cudaDeviceSynchronize();auto begin=std::chrono::steady_clock::now();for(int repeat=0;repeat<3;repeat++)pipeline();cudaDeviceSynchronize();double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()/3;cudaMemcpy(got.data(),buf,bytes,cudaMemcpyDeviceToHost);cudaMemcpy(&err,cnt,4,cudaMemcpyDeviceToHost);bool ok=got==expected&&err==0&&inject;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/trial_"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(got.data(),4,words,f);fwrite(&err,4,1,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":524289,\"max_abs_error\":%d,\"timing_region\":\"native_test5_init_move_check_sync_chain_mean\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":524289,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",ok?0:1,path.c_str());if(!ok)return 8;}
}
