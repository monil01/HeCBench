#pragma once
#include <vector>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstring>
#include <algorithm>
struct RLCase{unsigned n,steps,np;double t,dt;std::vector<double>s,p,first,final,gate;};
template<class T>void rlread(std::ifstream&f,T&x){f.read((char*)&x,sizeof(x));if(!f)throw std::runtime_error("short/missingfixture/oracle");}
inline std::vector<RLCase>rlcases(const std::string&root,bool oracle){std::ifstream f(root+"/fixtures_compact.bin",std::ios::binary);unsigned nc;rlread(f,nc);if(nc!=3)throw std::runtime_error("casecount");std::vector<RLCase>cs;for(unsigned c=0;c<nc;c++){RLCase x;rlread(f,x.n);rlread(f,x.steps);rlread(f,x.np);rlread(f,x.t);rlread(f,x.dt);if(!x.n||!x.np||x.np>x.n)throw std::runtime_error("caseshape");x.s.resize(19*x.np);x.p.resize(53*x.np);for(unsigned k=0;k<x.np;k++){for(unsigned j=0;j<19;j++)rlread(f,x.s[j*x.np+k]);for(unsigned j=0;j<53;j++)rlread(f,x.p[j*x.np+k]);}cs.push_back(x);}if(f.peek()!=EOF)throw std::runtime_error("fixturetrailing");if(oracle){std::ifstream o(root+"/oracle_compact.bin",std::ios::binary),g(root+"/independent_gates.bin",std::ios::binary);unsigned no,ng;rlread(o,no);rlread(g,ng);if(no!=nc||ng!=nc)throw std::runtime_error("oraclecases");for(auto&x:cs){unsigned np;rlread(o,np);if(np!=x.np)throw std::runtime_error("oraclepatterns");x.first.resize(19*np);x.final.resize(19*np);for(double&v:x.first)rlread(o,v);for(double&v:x.final)rlread(o,v);rlread(g,np);if(np!=x.np)throw std::runtime_error("gatepatterns");x.gate.resize(13*np);for(double&v:x.gate)rlread(g,v);}if(o.peek()!=EOF||g.peek()!=EOF)throw std::runtime_error("oracletrailing");}return cs;}
inline std::vector<double>rlexpand(const std::vector<double>&a,unsigned fields,unsigned n,unsigned np){std::vector<double>v(fields*n);for(unsigned f=0;f<fields;f++)for(unsigned i=0;i<n;i++)v[f*n+i]=a[f*np+i%np];return v;}
