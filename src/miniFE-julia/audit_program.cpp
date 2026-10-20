// CPU-only replay of retained canonical actual inputs and every full final field.
#include "q1_oracle.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdio>
template<class T>static std::vector<T>take(std::ifstream&f,size_t n){std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));if((size_t)f.gcount()!=n*sizeof(T))throw std::runtime_error("Truncated full array");return v;}
static Q1Input load(const std::string&dir,int dim){std::ifstream f(dir+"/actual_input_"+std::to_string(dim)+".bin",std::ios::binary);auto h=take<uint64_t>(f,4);if(h[0]!=0x4d4645494e505554ULL||h[1]!=(uint64_t)dim||h[2]!=(uint64_t)(dim+1)*(dim+1)*(dim+1))throw std::runtime_error("Actual input header mismatch");Q1Input v;v.dim=dim;v.n=h[2];v.nz=h[3];v.rp=take<int>(f,v.n+1);v.ci=take<int>(f,v.nz);v.A=take<double>(f,v.nz);v.b=take<double>(f,v.n);v.x=take<double>(f,v.n);v.r=take<double>(f,v.n);v.p=take<double>(f,v.n);v.Ap=take<double>(f,v.n);if(f.peek()!=std::char_traits<char>::eof()||!q1_review_matrix(v).pass)throw std::runtime_error("Full canonical Q1 matrix/RHS gate failed");return v;}
int main(int argc,char**argv){if(argc<4)return 2;try{
 std::vector<Q1Input>cases;for(int d:{2,16,128})cases.push_back(load(argv[1],d));std::ofstream out(argv[2]);out<<"{\"artifacts\":[";bool firstArtifact=true;bool all=true;
 for(int ai=3;ai<argc;ai++){
  std::ifstream f(argv[ai],std::ios::binary);
  for(auto&v:cases){auto h=take<double>(f,3);if(h[0]!=v.dim||h[1]!=v.n||h[2]!=v.nz)throw std::runtime_error("Flat exact prefix header mismatch");auto b=take<double>(f,v.n),x=take<double>(f,v.n);if(memcmp(b.data(),v.b.data(),8*v.n)||memcmp(x.data(),v.x.data(),8*v.n))throw std::runtime_error("Full actual b/x0 prefix differs from retained input bits");}
  out<<(firstArtifact?"":",")<<"{\"path\":\""<<argv[ai]<<"\",\"cases\":[";bool firstCase=true;
  for(auto&v:cases){auto h=take<double>(f,3);if(h[0]!=v.dim||h[1]!=v.n||h[2]!=v.nz)throw std::runtime_error("Flat final-state header mismatch");auto x=take<double>(f,v.n),r=take<double>(f,v.n),p=take<double>(f,v.n),Ap=take<double>(f,v.n),stored=take<double>(f,v.n);auto review=q1_review_solution(v,x);double difference=0;bool finite=true;for(auto*a:{&x,&r,&p,&Ap,&stored})for(double z:*a)finite&=std::isfinite(z);for(int i=0;i<v.n;i++)difference=std::max(difference,std::abs(stored[i]-review.residual[i]));bool pass=review.pass&&finite&&difference<=1e-18;all&=pass;out.precision(17);out<<(firstCase?"":",")<<"{\"dim\":"<<v.dim<<",\"nodes\":"<<v.n<<",\"true_relative_residual\":"<<review.relative_true_residual<<",\"max_boundary_error\":"<<review.max_boundary_abs<<",\"center\":"<<review.center<<",\"max_stored_vs_recomputed_residual_difference\":"<<difference<<",\"pass\":"<<(pass?"true":"false")<<"}";firstCase=false;}
  if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Unexpected extra artifact values");out<<"]}";firstArtifact=false;
 }out<<"],\"all_pass\":"<<(all?"true":"false")<<",\"scope\":\"CPU replay of the same independent Q1/long-double oracle, not a second mathematical oracle implementation or new GPU trial.\"}\n";return all?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 3;}}
