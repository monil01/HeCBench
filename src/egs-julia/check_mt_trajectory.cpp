// Independent serial canonical recurrence. No CUDA/source kernel inclusion.
#include <array>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <string>
using U=uint32_t;
std::vector<U>read(const std::string&p,size_t n){FILE*f=fopen(p.c_str(),"rb");assert(f);std::vector<U>v(n);assert(fread(v.data(),4,n,f)==n);assert(fgetc(f)==EOF);fclose(f);return v;}
int main(int argc,char**argv){assert(argc>=3);std::string d=argv[1];auto par=read(d+"/initial_mt_params.bin",4),tab=read(d+"/initial_mt_tables.bin",32),init=read(d+"/initial_mt_state.bin",128);std::array<U,101>s;for(int i=0;i<101;i++)s[i]=init[i];U M=par[0],sh1=par[1],sh2=par[2],mask=par[3];assert(M>0&&M<101&&101-M>=32);printf("{\"independent_scalar_recurrence\":true,\"phases\":[");for(int p=2;p<argc;p++){std::string name=argv[p];auto target=read(d+"/"+name+"_mt_state.bin",128);for(int j=101;j<128;j++)assert(target[j]==init[j]);int generation=0;bool matched=false;for(generation=1;generation<=5000000;generation++){for(int i=0;i<101;i++){U x=(s[i]&mask)^s[(i+1)%101];x^=x<<sh1;x^=s[(i+M)%101]>>sh2;x^=tab[x&15];s[i]=x;}if(s[0]!=target[0]||s[1]!=target[1]||s[2]!=target[2])continue;matched=true;for(int i=0;i<101;i++)matched&=s[i]==target[i];if(matched)break;}assert(matched);printf("%s{\"phase\":\"%s\",\"canonical_generations_since_previous_capture\":%d,\"all101_state_words_exact\":true,\"unused27_words_unchanged\":true}",p==2?"":",",name.c_str(),generation);}printf("],\"all_pass\":true,\"scope\":\"State membership in scalar MTGP recurrence, not exact random drawcount/statistical quality/full transport validation\"}\n");}
