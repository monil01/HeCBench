#define NPTS 128
#define main legacy_main
#include "study_native_source.cuh"
#undef main
#include <fstream>
#include <iomanip>
int main(){std::vector<float> A(16384),B(16384);float *a,*b,*o;int *ix;cudaMalloc(&a,65536);cudaMalloc(&b,65536);cudaMalloc(&o,512);cudaMalloc(&ix,512);std::vector<double> prefix;std::vector<std::vector<float>> AA,BB;for(int c=0;c<2;c++){for(int i=0;i<128;i++)for(int d=0;d<128;d++){A[i*128+d]=c==0?float((i*13+d*7)%17-8)/16:(i==0?0:(d==i?1:0));B[i*128+d]=c==0?float((i*19+d*11)%23-11)/16:((i==0||i==127)?1:-1);}AA.push_back(A);BB.push_back(B);prefix.insert(prefix.end(),A.begin(),A.end());prefix.insert(prefix.end(),B.begin(),B.end());for(int x:{128,128,3})prefix.push_back(x);}
auto run=[&](int c,int k,std::vector<double>&out){cudaMemcpy(a,AA[c].data(),65536,cudaMemcpyHostToDevice);cudaMemcpy(b,BB[c].data(),65536,cudaMemcpyHostToDevice);cudaMemset(o,0,512);cudaMemset(ix,255,512);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int rep=0;rep<3;rep++){switch(k){
case 1:Match1<<<1,dim3(128,1)>>>(a,b,o,ix);break;
case 2:Match2<<<8,dim3(16,16)>>>(a,b,o,ix);break;
case 3:Match3<<<8,dim3(16,16)>>>(a,b,o,ix);break;
case 4:Match4<<<8,dim3(16,16)>>>(a,b,o,ix);break;
case 5:Match5<<<8,dim3(16,16)>>>(a,b,o,ix);break;
case 6:Match6<<<8,dim3(16,16)>>>(a,b,o,ix);break;
case 7:Match7<<<4,dim3(32,8)>>>(a,b,o,ix);break;
case 8:Match8<<<4,dim3(32,8)>>>(a,b,o,ix);break;
case 9:Match9<<<4,dim3(32,4)>>>(a,b,o,ix);break;
case 10:Match10<<<4,dim3(32,8)>>>(a,b,o,ix);break;
}}cudaDeviceSynchronize();double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/3;std::vector<float> scores(128);std::vector<int> indices(128);cudaMemcpy(scores.data(),o,512,cudaMemcpyDeviceToHost);cudaMemcpy(indices.data(),ix,512,cudaMemcpyDeviceToHost);bool ok=cudaGetLastError()==cudaSuccess;for(int i=0;i<128;i++){long best=0;int winner=-1;int groups=k<=5?1:k==6?4:8;int tile=k<=5?128:k==6?16:32;for(int g=0;g<groups;g++)for(int bp=0;bp<128;bp+=tile)for(int q=0;q<(k<=5?128:4);q++){int j=bp+(k<=5?q:4*g+q);long dot=0;for(int d=0;d<128;d++)dot+=long(AA[c][i*128+d]*16)*long(BB[c][j*128+d]*16);if(dot>best){best=dot;winner=j;}}ok&=scores[i]==float(best)/256&&indices[i]==winner;}out.insert(out.end(),scores.begin(),scores.end());out.insert(out.end(),indices.begin(),indices.end());return std::make_pair(t,ok);};std::vector<double> tmp;for(int c=0;c<2;c++)for(int k=1;k<=10;k++)run(c,k,tmp);for(int trial=1;trial<=5;trial++){auto out=prefix;double elapsed=0;bool ok=true;std::cout<<"STUDY_COMPONENT_SECONDS ";for(int c=0;c<2;c++)for(int k=1;k<=10;k++){auto[t,p]=run(c,k,out);elapsed+=t;ok&=p;std::cout<<std::setprecision(17)<<t<<",";}std::cout<<std::endl;std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<std::setprecision(17)<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":0,\"timing_region\":\"sum20_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}}