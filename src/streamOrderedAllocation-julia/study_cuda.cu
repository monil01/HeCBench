#define main original_main
#include "../streamOrderedAllocation-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <dlfcn.h>
#undef repeat
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAAPI failure %s\n",cudaGetErrorString(e));exit(2);}}
std::vector<double> readall(const char *f){std::ifstream in(f,std::ios::binary|std::ios::ate);if(!in)exit(3);size_t n=in.tellg();if(n%8)exit(3);in.seekg(0);std::vector<double> v(n/8);in.read((char*)v.data(),n);if(!in)exit(3);return v;}
struct Case {int n;std::vector<float>a,b,out,ea,eb;};
int main(){int supported=0;ck(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,0));if(!supported){fprintf(stderr,"Memory pools unsupported on actual device\n");return 4;}int ver;ck(cudaRuntimeGetVersion(&ver));Dl_info info{};auto sym=dlsym(RTLD_DEFAULT,"cudaMallocAsync");if(!sym||!dladdr(sym,&info))return 5;printf("CUDART_PROVENANCE version=%d path=%s\n",ver,info.dli_fname);
 auto pre=readall(FIXTURE),gold=readall(ORACLE);if(pre.size()!=9229||gold.size()!=1410660)return 3;size_t pos=1;std::vector<Case> cases;
 for(int ci=0;ci<4;ci++){Case c;c.n=pre[pos++];if(pre[pos++]!=100)return 3;c.a.resize(c.n);c.b.resize(c.n);c.ea.resize(c.n);c.eb.resize(c.n);c.out.resize(100*c.n);for(auto&v:c.a)v=pre[pos++];for(auto&v:c.b)v=pre[pos++];cases.push_back(c);}if(pos!=pre.size())return 3;
 cudaMemPool_t pool;ck(cudaDeviceGetDefaultMemPool(&pool,0));uint64_t original;ck(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&original));
 auto execute=[&](Case&c,int mode){cudaStream_t stream=0;if(mode)ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));if(mode){uint64_t threshold=mode==1?0:UINT64_MAX;ck(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold));}ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
 for(int it=0;it<100;it++){float *a,*b,*out;size_t bytes=c.n*4;if(mode){ck(cudaMallocAsync(&a,bytes,stream));ck(cudaMallocAsync(&b,bytes,stream));ck(cudaMallocAsync(&out,bytes,stream));ck(cudaMemcpyAsync(a,c.a.data(),bytes,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(b,c.b.data(),bytes,cudaMemcpyHostToDevice,stream));}else{ck(cudaMalloc(&a,bytes));ck(cudaMalloc(&b,bytes));ck(cudaMalloc(&out,bytes));ck(cudaMemcpy(a,c.a.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(b,c.b.data(),bytes,cudaMemcpyHostToDevice));}
 vectorAddGPU<<<(c.n+255)/256,256,0,stream>>>(a,b,out,c.n);ck(cudaGetLastError());
 if(it==99){if(mode){ck(cudaMemcpyAsync(c.ea.data(),a,bytes,cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(c.eb.data(),b,bytes,cudaMemcpyDeviceToHost,stream));}else{ck(cudaMemcpy(c.ea.data(),a,bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(c.eb.data(),b,bytes,cudaMemcpyDeviceToHost));}}
 if(mode){ck(cudaFreeAsync(a,stream));ck(cudaFreeAsync(b,stream));ck(cudaMemcpyAsync(c.out.data()+it*c.n,out,bytes,cudaMemcpyDeviceToHost,stream));ck(cudaFreeAsync(out,stream));ck(cudaStreamSynchronize(stream));}else{ck(cudaFree(a));ck(cudaFree(b));ck(cudaMemcpy(c.out.data()+it*c.n,out,bytes,cudaMemcpyDeviceToHost));ck(cudaFree(out));}}
 auto stop=std::chrono::steady_clock::now();double t=std::chrono::duration<double>(stop-start).count();if(mode)ck(cudaStreamDestroy(stream));return t;};
 for(auto&c:cases)for(int mode=0;mode<3;mode++)execute(c,mode);
 for(int trial=1;trial<=5;trial++){auto art=pre;std::vector<double>times;bool ok=true;
 for(auto&c:cases)for(int mode=0;mode<3;mode++){times.push_back(execute(c,mode));for(auto x:c.out)art.push_back(x);for(auto x:c.ea)art.push_back(x);for(auto x:c.eb)art.push_back(x);ok&=c.a==c.ea&&c.b==c.eb;}
 if(art.size()!=pre.size()+gold.size())return 6;for(size_t i=0;i<gold.size();i++)ok&=art[pre.size()+i]==gold[i]&&std::isfinite(art[pre.size()+i]);std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)art.data(),art.size()*8);f.close();if(!f)return 7;double total=0;for(auto x:times)total+=x;
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum12_source_100repeat_allocator_copy_add_free_pipelines_plus_final_inputecho\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,total,ok?"true":"false",art.size(),path.c_str(),art.size());for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"authored_device_kernels\":1,\"GPU_guards_present\":false,\"all_allocation_copy_free_API_status_checked\":true}\n");}
 ck(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&original));}
