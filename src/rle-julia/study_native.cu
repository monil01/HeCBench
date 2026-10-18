#include <cstdint>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#define main original_rle_cli_main
#include "../rle-cuda/main.cu"
#undef main
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<class T>std::vector<T> read(std::ifstream&f,size_t n){std::vector<T>a(n);f.read((char*)a.data(),n*sizeof(T));if(!f)exit(3);return a;}
int* upload(std::vector<int>a){int*d;CK(cudaMalloc(&d,a.size()*4));CK(cudaMemcpy(d,a.data(),a.size()*4,cudaMemcpyHostToDevice));return d;}
void capture(std::vector<int>&a,int*d,size_t n){size_t b=a.size();a.resize(b+n);CK(cudaMemcpy(a.data()+b,d,n*4,cudaMemcpyDeviceToHost));}
struct Case{int n,kind;int*in,*key,*len,*count;void*temp;size_t bytes;};
void encode(Case&c,int r){if(c.kind)CK(Dispatch(r,c.temp,c.bytes,c.in,c.key,c.len,c.count,c.n));else CK(Dispatch(r,c.temp,c.bytes,ConstantInputIterator<int,int>(1),c.key,c.len,c.count,c.n));}
int main(){std::ifstream f("fixture.bin",std::ios::binary);int nc=read<int>(f,1)[0];std::vector<Case>cases;std::vector<int>prefix;for(int z=0;z<nc;z++){auto h=read<int>(f,4),in=read<int>(f,h[0]);int n=h[0],kind=h[1];std::vector<int>buf(n+16,0),count(17,1597463007);std::fill(buf.end()-16,buf.end(),1597463007);count[0]=0;Case c{n,kind,kind?upload(in):nullptr,upload(buf),upload(buf),upload(count),nullptr,0};if(kind)capture(prefix,c.in,n);encode(c,1);CK(cudaMalloc(&c.temp,c.bytes));encode(c,1);cases.push_back(c);}CK(cudaDeviceSynchronize());std::ifstream ef("expected_i32.bin",std::ios::binary);ef.seekg(0,std::ios::end);size_t es=ef.tellg();ef.seekg(0);auto expected=read<int>(ef,es/4);bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<int>data=prefix;std::vector<double>times;double total=0;for(auto&c:cases){CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();encode(c,1000);double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/1000.;total+=time;times.push_back(time);capture(data,c.key,c.n+16);capture(data,c.len,c.n+16);capture(data,c.count,17);}bool ok=std::equal(expected.begin(),expected.end(),data.begin()+prefix.size());all&=ok;size_t bad=0;for(size_t j=0;j<expected.size();j++)bad+=expected[j]!=data[prefix.size()+j];const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/rle_trial%d.bin",dir?dir:".",trial);std::ofstream of(file,std::ios::binary);of.write((char*)data.data(),data.size()*4);of.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%lu,\"mismatches\":%lu,\"max_abs_error\":null,\"max_rel_error\":null,\"jit_excluded\":true,\"component_seconds\":[",trial,total,ok?"true":"false",expected.size(),bad);for(size_t j=0;j<times.size();j++)printf("%s%.12g",j?",":"",times[j]);printf("],\"timing_region\":\"sum7_source1000Encode_means_each_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%lu,\"byte_order\":\"little\"}}\n",file,data.size());fflush(stdout);}puts(all?"PASS":"FAIL");return all?0:1;}
