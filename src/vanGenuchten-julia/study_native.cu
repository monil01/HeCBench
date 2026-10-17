#define main legacy_main
#include "vanGenuchten-cuda/main.cu"
#undef main
#include <vector>
int main(){int size=4096,repeat=10;std::vector<double>s(size),p(size),c(size),t(size),k(size),cr(size),tr(size),kr(size),out;
for(int i=0;i<size;i++){s[i]=1e-6+(1.-1e-6)*i/size;p[i]=-100.+101.*i/size;}reference(s.data(),p.data(),cr.data(),tr.data(),kr.data(),size);
double *ds,*dp,*dc,*dt,*dk;cudaMalloc(&ds,size*8);cudaMalloc(&dp,size*8);cudaMalloc(&dc,size*8);cudaMalloc(&dt,size*8);cudaMalloc(&dk,size*8);cudaMemcpy(ds,s.data(),size*8,cudaMemcpyHostToDevice);cudaMemcpy(dp,p.data(),size*8,cudaMemcpyHostToDevice);
vanGenuchten<<<(size+255)/256,256>>>(ds,dp,dc,dt,dk,size);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
for(int trial=1;trial<=5;trial++){cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeat;j++)vanGenuchten<<<(size+255)/256,256>>>(ds,dp,dc,dt,dk,size);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;
cudaMemcpy(c.data(),dc,size*8,cudaMemcpyDeviceToHost);cudaMemcpy(t.data(),dt,size*8,cudaMemcpyDeviceToHost);cudaMemcpy(k.data(),dk,size*8,cudaMemcpyDeviceToHost);double error=0;bool correct=true;for(int i=0;i<size;i++){for(auto v:{c[i],t[i],k[i]})correct &=std::isfinite(v);error=fmax(error,fabs(c[i]-cr[i]));error=fmax(error,fabs(t[i]-tr[i]));error=fmax(error,fabs(k[i]-kr[i]));}correct &=error<=1e-3;out=c;out.insert(out.end(),t.begin(),t.end());out.insert(out.end(),k.begin(),k.end());char file[32];snprintf(file,sizeof(file),"trial_%d.bin",trial);FILE*f=fopen(file,"wb");fwrite(out.data(),8,out.size(),f);fclose(f);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.14g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.14g,\"timing_region\":\"synchronized_kernel_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,correct?"true":"false",size*3,error,repeat,file,size*3);if(!correct)return 1;}cudaFree(ds);cudaFree(dp);cudaFree(dc);cudaFree(dt);cudaFree(dk);}
