#pragma once
// Full lossless raw-frame CAS. Every input byte is hashed; no numerical summaries substitute data.
#include <openssl/sha.h>
#include <zlib.h>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <unistd.h>
inline void study_need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
inline std::string study_quote(const std::string&s){std::string o="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){o+='\\';o+=c;}else if(c=='\n')o+="\\n";else{study_need(c>=32,"JSON control byte");o+=c;}}return o+"\"";}
inline std::string study_sha(const unsigned char*p,size_t n){unsigned char h[SHA256_DIGEST_LENGTH];SHA256(p,n,h);std::ostringstream s;for(auto c:h)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return s.str();}
struct StudyCAS{std::filesystem::path dir;std::set<std::string>seen;explicit StudyCAS(std::filesystem::path d):dir(std::move(d)){std::filesystem::create_directories(dir);}std::string put(const std::vector<unsigned char>&v){std::string sha=study_sha(v.data(),v.size());auto path=dir/(sha+".bin.gz");if(!std::filesystem::exists(path)){auto temp=path;temp+=".tmp."+std::to_string(getpid());gzFile f=gzopen(temp.c_str(),"wb1");study_need(f,"gzip create");size_t off=0;while(off<v.size()){unsigned n=std::min<size_t>(1<<20,v.size()-off);study_need(gzwrite(f,v.data()+off,n)==int(n),"gzip write");off+=n;}study_need(gzclose(f)==Z_OK,"gzip close");std::filesystem::rename(temp,path);}seen.insert(sha);return sha;}};
struct StudyCase{uint64_t vectors,fields;int iterations;};
inline std::vector<StudyCase>study_cases(){return{{4000,90000,10},{17,31,3},{2,1,2}};}
inline uint64_t study_stride(uint64_t r){return 1+r%499;}
inline uint64_t study_gcd(uint64_t a,uint64_t b){while(b){uint64_t t=a%b;a=b;b=t;}return a;}
inline uint64_t study_intersection(uint64_t r,uint64_t c,uint64_t fields){uint64_t a=study_stride(r),b=study_stride(c),l=(a/study_gcd(a,b))*b;return 1+(fields-1)/l;}
inline uint64_t study_hash_element(uint64_t value,uint64_t r,uint64_t c){return 1+((value*r*c)&((uint64_t(1)<<40)-1));}
inline uint64_t study_global_hash(const StudyCase&c){uint64_t h=0;for(uint64_t j=1;j<c.vectors;j++)for(uint64_t i=0;i<j;i++)h+=study_hash_element(study_intersection(i,j,c.fields),i,j);return h*uint64_t(c.iterations);}
inline std::vector<float>study_expected_input(uint64_t rows,uint64_t cols,uint64_t ld,uint64_t alloccols,uint64_t base){std::vector<float>v(ld*alloccols,0);for(uint64_t c=0;c<cols;c++){float pattern[499];for(uint64_t s=1;s<=499;s++)pattern[s-1]=c%s?0.f:1.f;uint64_t k=base%499;for(uint64_t r=0;r<rows;r++){v[r+ld*c]=pattern[k];if(++k==499)k=0;}}return v;}
inline void study_check_C(const float*p,uint64_t m,uint64_t ld,uint64_t ac,uint64_t left,uint64_t right,uint64_t fields){std::vector<float>table(499*499);for(uint64_t c=0;c<499;c++)for(uint64_t r=0;r<499;r++)table[r+499*c]=float(study_intersection(r,c,fields));for(uint64_t c=0;c<ac;c++){uint64_t ri=left%499,ci=(c+right)%499;for(uint64_t r=0;r<ld;r++){float want=(r<m&&c<m)?table[ri+499*ci]:0.f;study_need(p[r+ld*c]==want,"independent complete SGEMM lcm/padding oracle");if(++ri==499)ri=0;}}}
