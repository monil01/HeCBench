// Execute the unchanged original SU parsing/aperture/CDP sorting implementation.
#include "../cmp-cuda/su_gather.hpp"
#include "../cmp-cuda/log.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio>
int main(int argc,char**argv){assert(argc==2);logger::verbosity_level(0);std::vector<double>out{2};for(int ci=0;ci<2;ci++){int nc=ci==0?5:9;std::string path=std::string(argv[1])+"/synthetic"+std::to_string(ci)+".su";su_gather gather(path,600,nc);int*counts;float *samples,*gx,*gy,*sx,*sy,*scalco,dt;gather.linearize(counts,samples,dt,gx,gy,sx,sy,scalco,nc);int ns=gather.ns(),ncdps=gather().size(),nt=gather.ttraces(),ntrs=gather.ntrs();float c0=std::stof("1.98e-7")*FACTOR,c1=std::stof("1.77e-6")*FACTOR,inc=(c1-c0)*(1.f/float(nc));dt=dt/1000000.f;float idt=1.f/dt,itau=std::stof(ci==0?"0.002":"0.0141");int tau=int(itau*idt)>0?int(itau*idt):0,w=2*tau+1;assert(w<=MAX_W);out.insert(out.end(),{double(nc),double(ns),double(ncdps),double(nt),double(ntrs),c0,inc,dt,idt,double(tau),double(w)});for(int i=0;i<ncdps;i++)out.push_back(counts[i]);for(float*p:{scalco,gx,gy,sx,sy})for(int i=0;i<nt;i++)out.push_back(p[i]);for(int i=0;i<nt*ns;i++)out.push_back(samples[i]);delete[]counts;delete[]samples;delete[]gx;delete[]gy;delete[]sx;delete[]sy;delete[]scalco;fprintf(stderr,"SOURCE_SU case%d ns%d nc%d CDPs%d traces%d max%d tau%d w%d\n",ci,ns,nc,ncdps,nt,ntrs,tau,w);}std::ofstream f("fixture.bin",std::ios::binary);f.write((char*)out.data(),out.size()*8);assert(f);}
