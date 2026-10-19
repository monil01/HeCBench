#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
struct STSGFixture {int nx=17,ny=13,nb=12,years=3,win=4;float cosyear=.8f,sampcorr=.5f;int snow=1;std::vector<short>raw;std::vector<unsigned char>qa;STSGFixture(){raw.resize(nx*ny*nb*years);qa.resize(raw.size());for(int y=0;y<years;y++)for(int k=0;k<nb;k++)for(int j=0;j<ny;j++)for(int i=0;i<nx;i++){int p=i+j*nx+k*nx*ny+y*nx*ny*nb;raw[p]=short(3500+1200*std::sin((k+1)*.5)+23*((i*3+j*7)%11)+65*y*((k%3)-1));qa[p]=0;}}};
