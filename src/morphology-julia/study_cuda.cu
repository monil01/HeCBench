// Original vHGW kernels, canonical plus nontrivial boundary-sensitive fixture.
#include "../morphology-cuda/morphology.cu"
#include <vector>
#include <algorithm>
void axis_cpu(std::vector<unsigned char>&out,const std::vector<unsigned char>&in,int W,int H,int sel,int op,int axis){int half=sel/2;for(int y=0;y<H;y++)for(int x=0;x<W;x++){int i=y*W+x,p=axis==1?x:y,limit=axis==1?W:H;if(p<half||p>=limit-half)out[i]=axis==1?0:(op==1?255:0);else{unsigned char acc=op==1?0:255;for(int k=-half;k<=half;k++){int j=axis==1?i+k:i+k*W;acc=op==1?std::max(acc,in[j]):std::min(acc,in[j]);}out[i]=acc;}}}
void cycle(unsigned char*a,unsigned char*b,int W,int H){dim3 hb(5,1),hg(W/5,H),vb(1,5),vg(W,H/5);cudaMemset(b,0,W*H);vhgw_horiz<MorphOpType::DILATE><<<hg,hb,20>>>(b,a,W,H,5);vhgw_vert<MorphOpType::DILATE><<<vg,vb,20>>>(a,b,W,H,5);cudaMemset(b,0,W*H);vhgw_horiz<MorphOpType::ERODE><<<hg,hb,20>>>(b,a,W,H,5);vhgw_vert<MorphOpType::ERODE><<<vg,vb,20>>>(a,b,W,H,5);}
int main(int argc,char**argv){int repeats=argc>1?atoi(argv[1]):10,W=510,H=510,n=W*H;if(repeats<=0)return 2;std::vector<std::vector<unsigned char>>inputs(2,std::vector<unsigned char>(n,0)),expected;inputs[0][(H/2-1)*W+(W/2-1)]=255;for(int i=0;i<n;i++)inputs[1][i]=(i*37+(i/W)*13+17)%256;
 for(auto input:inputs){auto tmp=input;for(int k=0;k<repeats;k++)for(int op=1;op<=2;op++){axis_cpu(tmp,input,W,H,5,op,1);axis_cpu(input,tmp,W,H,5,op,2);}expected.push_back(input);}
 unsigned char*a,*b;cudaMalloc(&a,n);cudaMalloc(&b,n);cudaMemcpy(a,inputs[0].data(),n,cudaMemcpyHostToDevice);cycle(a,b,W,H);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){std::vector<unsigned char>allgot;double total=0;int ma=0;bool ok=true;
  for(int c=0;c<2;c++){cudaMemcpy(a,inputs[c].data(),n,cudaMemcpyHostToDevice);cudaMemset(b,0,n);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int k=0;k<repeats;k++)cycle(a,b,W,H);auto status=cudaDeviceSynchronize();total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<unsigned char>got(n);cudaMemcpy(got.data(),a,n,cudaMemcpyDeviceToHost);ok &=status==cudaSuccess;for(int i=0;i<n;i++){int e=abs(int(got[i])-expected[c][i]);ma=std::max(ma,e);ok &=e==0;}allgot.insert(allgot.end(),got.begin(),got.end());}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"morphology_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(allgot.data(),1,allgot.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"closing_cycle_four_separable_kernels_two_temporary_resets\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,total/(repeats*2),ok?"true":"false",allgot.size(),ma,repeats,file,allgot.size());
 }
 cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;}
