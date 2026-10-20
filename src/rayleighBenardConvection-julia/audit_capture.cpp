// CPU-only full artifact/prefix/oracle replay. No CUDA or source solver helpers.
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
template<class T>std::vector<T>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::string("missing ")+p;size_t n=f.tellg();if(n%sizeof(T))throw std::string("size");std::vector<T>v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);if(!f)throw std::string("short");return v;}
int main(int n,char**a){if(n!=5)return 2;auto input=load<float>(a[1]);auto gold=load<double>(a[2]);auto budget=load<double>(a[3]);auto got=load<float>(a[4]);if(gold.size()!=budget.size()||got.size()!=input.size()+gold.size())return 3;bool ok=memcmp(got.data(),input.data(),input.size()*4)==0;double err=0,ratio=0;for(size_t i=0;i<gold.size();i++){double e=std::abs(double(got[input.size()+i])-gold[i]);if(!std::isfinite(got[input.size()+i])||!std::isfinite(gold[i])||e>budget[i])ok=false;err=std::max(err,e);ratio=std::max(ratio,e/budget[i]);}printf("{\"pass\":%s,\"actual_input_values_byte_exact\":%zu,\"full_output_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_field_budget_ratio\":%.17g}\n",ok?"true":"false",input.size(),gold.size(),err,ratio);return ok?0:4;}
