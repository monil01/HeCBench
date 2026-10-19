#pragma once
#include <vector>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cstring>
constexpr int FGUARD=1234567890,FG=64,FINF=1073741823;
struct FCase {int n,e,u,r;std::vector<int>idx,nei,wt,oracle;};
static void f_read(std::ifstream&f,void*p,size_t bytes){if(!f.read(static_cast<char*>(p),bytes))throw std::runtime_error("Missing/truncated independent fixture");}
static std::vector<FCase> f_cases(){std::ifstream f("fixtures.bin",std::ios::binary);uint32_t count;f_read(f,&count,4);if(count!=7)throw std::runtime_error("case count");std::vector<FCase>v;for(uint32_t i=0;i<count;i++){FCase c;f_read(f,&c.n,4);f_read(f,&c.e,4);f_read(f,&c.u,4);f_read(f,&c.r,4);if(c.n<1||c.u<64||c.u>1088||c.u%64||c.n>c.u||c.e<1||c.r!=3)throw std::runtime_error("fixture geometry");c.idx.resize(c.n+1);c.nei.resize(c.e);c.wt.resize(c.e);c.oracle.resize(c.u*c.u);for(auto*a:{&c.idx,&c.nei,&c.wt,&c.oracle})f_read(f,a->data(),a->size()*4);if(c.idx.front()!=0||c.idx.back()!=c.e)throw std::runtime_error("CSR bounds");v.push_back(std::move(c));}char tail;if(f.read(&tail,1))throw std::runtime_error("fixture trailingdata");return v;}
static void f_append(std::vector<int>&v,const std::vector<int>&a){v.insert(v.end(),a.begin(),a.end());}
static void f_assert_equal(const std::vector<int>&a,const std::vector<int>&b,const char*name){if(a!=b){for(size_t k=0;k<std::min(a.size(),b.size());k++)if(a[k]!=b[k]){fprintf(stderr,"%s index%zu actual%d expected%d\n",name,k,a[k],b[k]);break;}throw std::runtime_error(name);}}
