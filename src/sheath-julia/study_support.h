#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cmath>
inline void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
struct Case{int n,steps,cold;std::vector<double>ions_x,ions_v,electrons_x,electrons_v;};
inline std::vector<Case>cases(){std::vector<Case>out;for(int k=0;k<3;k++){std::ifstream f("data/case"+std::to_string(k)+".bin",std::ios::binary);Case c;f.read((char*)&c.n,4);f.read((char*)&c.steps,4);f.read((char*)&c.cold,4);need(f.good()&&c.n>0&&c.steps>0,"fixture header");c.ions_x.resize(c.n);c.ions_v.resize(c.n);c.electrons_x.resize(c.n);c.electrons_v.resize(c.n);for(int s=0;s<2;s++)for(int j=0;j<c.n;j++){double x,v;f.read((char*)&x,8);f.read((char*)&v,8);need(f.good()&&std::isfinite(x)&&std::isfinite(v),"fixture body");(s?c.electrons_x:c.ions_x)[j]=x;(s?c.electrons_v:c.ions_v)[j]=v;}need(f.peek()==EOF,"fixture trailingbytes");out.push_back(std::move(c));}return out;}
extern "C" {int witness_deposition(const void*,int,const float*,const float*,double);int witness_field(const float*,const float*,const double*,const double*,const double*,int);int witness_motion(const void*,const void*,int,const double*,double,int);void witness_report();}
