#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <chrono>
#define main historical_main
#include "../intrinsics-simd-cuda/main.cu"
#undef main
__global__ void individual_probe(const unsigned* input,unsigned* output,int count,int p){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;unsigned a=input[3*i],b=input[3*i+1],c=input[3*i+2],r=0;bool pred=false,pred_hi=false,pred_lo=false;
 switch(i/p){
 case 0:r=__vabs2(a);break;
 case 1:r=__vabs4(a);break;
 case 2:r=__vabsdiffs2(a, b);break;
 case 3:r=__vabsdiffs4(a, b);break;
 case 4:r=__vabsdiffu2(a, b);break;
 case 5:r=__vabsdiffu4(a, b);break;
 case 6:r=__vabsss2(a);break;
 case 7:r=__vabsss4(a);break;
 case 8:r=__vadd2(a, b);break;
 case 9:r=__vadd4(a, b);break;
 case 10:r=__vaddss2(a, b);break;
 case 11:r=__vaddss4(a, b);break;
 case 12:r=__vaddus2(a, b);break;
 case 13:r=__vaddus4(a, b);break;
 case 14:r=__vavgs2(a, b);break;
 case 15:r=__vavgs4(a, b);break;
 case 16:r=__vavgu2(a, b);break;
 case 17:r=__vavgu4(a, b);break;
 case 18:r=__vcmpeq2(a, b);break;
 case 19:r=__vcmpeq4(a, b);break;
 case 20:r=__vcmpges2(a, b);break;
 case 21:r=__vcmpges4(a, b);break;
 case 22:r=__vcmpgeu2(a, b);break;
 case 23:r=__vcmpgeu4(a, b);break;
 case 24:r=__vcmpgts2(a, b);break;
 case 25:r=__vcmpgts4(a, b);break;
 case 26:r=__vcmpgtu2(a, b);break;
 case 27:r=__vcmpgtu4(a, b);break;
 case 28:r=__vcmples2(a, b);break;
 case 29:r=__vcmples4(a, b);break;
 case 30:r=__vcmpleu2(a, b);break;
 case 31:r=__vcmpleu4(a, b);break;
 case 32:r=__vcmplts2(a, b);break;
 case 33:r=__vcmplts4(a, b);break;
 case 34:r=__vcmpltu2(a, b);break;
 case 35:r=__vcmpltu4(a, b);break;
 case 36:r=__vcmpne2(a, b);break;
 case 37:r=__vcmpne4(a, b);break;
 case 38:r=__vhaddu2(a, b);break;
 case 39:r=__vhaddu4(a, b);break;
 case 40:r=__viaddmax_s16x2(a, b, c);break;
 case 41:r=__viaddmax_s16x2_relu(a, b, c);break;
 case 42:r=__viaddmax_s32(a, b, c);break;
 case 43:r=__viaddmax_s32_relu(a, b, c);break;
 case 44:r=__viaddmax_u16x2(a, b, c);break;
 case 45:r=__viaddmax_u32(a, b, c);break;
 case 46:r=__viaddmin_s16x2(a, b, c);break;
 case 47:r=__viaddmin_s16x2_relu(a, b, c);break;
 case 48:r=__viaddmin_s32(a, b, c);break;
 case 49:r=__viaddmin_s32_relu(a, b, c);break;
 case 50:r=__viaddmin_u16x2(a, b, c);break;
 case 51:r=__viaddmin_u32(a, b, c);break;
 case 52:r=__vibmax_s16x2(a, b, &pred_hi, &pred_lo);break;
 case 53:r=__vibmax_s32(a, b, &pred);break;
 case 54:r=__vibmax_u16x2(a, b, &pred_hi, &pred_lo);break;
 case 55:r=__vibmax_u32(a, b, &pred);break;
 case 56:r=__vibmin_s16x2(a, b, &pred_hi, &pred_lo);break;
 case 57:r=__vibmin_s32(a, b, &pred);break;
 case 58:r=__vibmin_u16x2(a, b, &pred_hi, &pred_lo);break;
 case 59:r=__vibmin_u32(a, b, &pred);break;
 case 60:r=__vimax3_s16x2(a, b, c);break;
 case 61:r=__vimax3_s16x2_relu(a, b, c);break;
 case 62:r=__vimax3_s32(a, b, c);break;
 case 63:r=__vimax3_s32_relu(a, b, c);break;
 case 64:r=__vimax3_u16x2(a, b, c);break;
 case 65:r=__vimax3_u32(a, b, c);break;
 case 66:r=__vimax_s16x2_relu(a, b);break;
 case 67:r=__vimax_s32_relu(a, b);break;
 case 68:r=__vimin3_s16x2(a, b, c);break;
 case 69:r=__vimin3_s16x2_relu(a, b, c);break;
 case 70:r=__vimin3_s32(a, b, c);break;
 case 71:r=__vimin3_s32_relu(a, b, c);break;
 case 72:r=__vimin3_u16x2(a, b, c);break;
 case 73:r=__vimin3_u32(a, b, c);break;
 case 74:r=__vimin_s16x2_relu(a, b);break;
 case 75:r=__vimin_s32_relu(a, b);break;
 case 76:r=__vmaxs2(a, b);break;
 case 77:r=__vmaxs4(a, b);break;
 case 78:r=__vmaxu2(a, b);break;
 case 79:r=__vmaxu4(a, b);break;
 case 80:r=__vmins2(a, b);break;
 case 81:r=__vmins4(a, b);break;
 case 82:r=__vminu2(a, b);break;
 case 83:r=__vminu4(a, b);break;
 case 84:r=__vneg2(a);break;
 case 85:r=__vneg4(a);break;
 case 86:r=__vnegss2(a);break;
 case 87:r=__vnegss4(a);break;
 case 88:r=__vsads2(a, b);break;
 case 89:r=__vsads4(a, b);break;
 case 90:r=__vsadu2(a, b);break;
 case 91:r=__vsadu4(a, b);break;
 case 92:r=__vseteq2(a, b);break;
 case 93:r=__vseteq4(a, b);break;
 case 94:r=__vsetges2(a, b);break;
 case 95:r=__vsetges4(a, b);break;
 case 96:r=__vsetgeu2(a, b);break;
 case 97:r=__vsetgeu4(a, b);break;
 case 98:r=__vsetgts2(a, b);break;
 case 99:r=__vsetgts4(a, b);break;
 case 100:r=__vsetgtu2(a, b);break;
 case 101:r=__vsetgtu4(a, b);break;
 case 102:r=__vsetles2(a, b);break;
 case 103:r=__vsetles4(a, b);break;
 case 104:r=__vsetleu2(a, b);break;
 case 105:r=__vsetleu4(a, b);break;
 case 106:r=__vsetlts2(a, b);break;
 case 107:r=__vsetlts4(a, b);break;
 case 108:r=__vsetltu2(a, b);break;
 case 109:r=__vsetltu4(a, b);break;
 case 110:r=__vsetne2(a, b);break;
 case 111:r=__vsetne4(a, b);break;
 case 112:r=__vsub2(a, b);break;
 case 113:r=__vsub4(a, b);break;
 case 114:r=__vsubss2(a, b);break;
 case 115:r=__vsubss4(a, b);break;
 case 116:r=__vsubus2(a, b);break;
 case 117:r=__vsubus4(a, b);break;
 }output[i]=r;
}
const unsigned SENT=0xdeadbeef;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<unsigned>load(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);size_t n=size_t(f.tellg());std::vector<unsigned>a(n/4);f.seekg(0);f.read((char*)a.data(),n);return a;}
struct Case{int n,reps;std::vector<unsigned>h;unsigned*in;unsigned*out;};
int main(){auto f=load(SIMD_FIXTURE),oracle=load(SIMD_ORACLE),po=load(SIMD_PROBE_ORACLE);size_t pos=0;std::vector<Case>cs;for(int j=0;j<3;j++){Case c;c.n=f[pos++];c.reps=f[pos++];c.h.assign(c.n+8,SENT);std::copy(f.begin()+pos,f.begin()+pos+c.n,c.h.begin()+4);pos+=c.n;ck(cudaMalloc(&c.in,(c.n+8)*4));ck(cudaMalloc(&c.out,(c.n+8)*4));ck(cudaMemcpy(c.in,c.h.data(),(c.n+8)*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(c.out,c.h.data(),(c.n+8)*4,cudaMemcpyHostToDevice));cs.push_back(c);}int ops=f[pos++],p=f[pos++],nt=ops*p;std::vector<unsigned>h(3*nt+8,SENT),z(nt+8,SENT);std::copy(f.begin()+pos,f.end(),h.begin()+4);unsigned*pin,*pout;ck(cudaMalloc(&pin,h.size()*4));ck(cudaMalloc(&pout,z.size()*4));ck(cudaMemcpy(pin,h.data(),h.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(pout,z.data(),z.size()*4,cudaMemcpyHostToDevice));
 for(auto&c:cs){simd_intrinsics<<<(c.n+255)/256,256>>>(c.n,c.in+4,c.out+4);ck(cudaDeviceSynchronize());}individual_probe<<<(nt+255)/256,256>>>(pin+4,pout+4,nt,p);ck(cudaDeviceSynchronize());
 for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double elapsed=0;size_t oi=0;std::vector<double>times;for(auto&c:cs){std::vector<unsigned>ib(c.n+8),ob(c.n+8,SENT);ck(cudaMemcpy(c.out,ob.data(),ob.size()*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<c.reps;rep++)simd_intrinsics<<<(c.n+255)/256,256>>>(c.n,c.in+4,c.out+4);ck(cudaDeviceSynchronize());double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/c.reps;elapsed+=t;times.push_back(t);ck(cudaMemcpy(ib.data(),c.in,ib.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ob.data(),c.out,ob.size()*4,cudaMemcpyDeviceToHost));ok&=ib==c.h;for(int k=0;k<4;k++)ok&=ob[k]==SENT&&ob[c.n+4+k]==SENT;for(int k=0;k<c.n;k++)ok&=ob[k+4]==oracle[oi++];payload.insert(payload.end(),ib.begin(),ib.end());payload.insert(payload.end(),ob.begin(),ob.end());}
 ck(cudaMemcpy(pout,z.data(),z.size()*4,cudaMemcpyHostToDevice));individual_probe<<<(nt+255)/256,256>>>(pin+4,pout+4,nt,p);ck(cudaDeviceSynchronize());std::vector<unsigned>ib(h.size()),ob(z.size());ck(cudaMemcpy(ib.data(),pin,ib.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ob.data(),pout,ob.size()*4,cudaMemcpyDeviceToHost));ok&=ib==h;for(int k=0;k<4;k++)ok&=ob[k]==SENT&&ob[nt+4+k]==SENT;for(int k=0;k<nt;k++)ok&=ob[k+4]==po[k];payload.insert(payload.end(),ib.begin(),ib.end());payload.insert(payload.end(),ob.begin(),ob.end());std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*4);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"timing_region\":\"sum3_simd_100launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":100,\"timed_kernel_calls\":300,\"untimed_individual_operation_results\":%d,\"case_times_seconds\":[",trial,elapsed,ok?"true":"false",payload.size(),ok?"0":"null",nt);for(size_t j=0;j<times.size();j++)printf("%s%.17g",j?",":"",times[j]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 3;}
 for(auto&c:cs){ck(cudaFree(c.in));ck(cudaFree(c.out));}ck(cudaFree(pin));ck(cudaFree(pout));}
