#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
constexpr int PW=61440,PG=256;
struct PC {int height,streams;float ppu;int seed,octaves;float lac,persistence;};
constexpr PC CASES[]={{512,1,250.0f,0,3,2.0f,0.5f},{512,2,250.0f,0,3,2.0f,0.5f},{512,4,250.0f,0,3,2.0f,0.5f},{512,8,250.0f,0,3,2.0f,0.5f},{512,16,250.0f,0,3,2.0f,0.5f},{512,32,250.0f,0,3,2.0f,0.5f},{16,1,1.0f,7,3,2.0f,0.5f},{48,2,250.0f,5,1,2.0f,0.5f}};
constexpr int PHASH[256]={151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};
constexpr uint64_t PREFIX=770720544,ARTIFACT=1541441088;
inline uint64_t pixels(int c){return uint64_t(PW)*CASES[c].height;}
inline std::vector<uint8_t> expected_constants(){std::vector<uint8_t>v(1088);std::memcpy(v.data(),PHASH,1024);float q=float(.70710678118654752440);float x[]={1,-1,0,0,q,-q,q,-q},y[]={0,0,1,-1,q,q,-q,-q};std::memcpy(v.data()+1024,x,32);std::memcpy(v.data()+1056,y,32);return v;}
inline std::vector<uint8_t> expected_params(int c){auto z=CASES[c];std::vector<uint8_t>v(36);float f[]={z.ppu,z.lac,z.persistence};std::memcpy(v.data(),&f[0],4);std::memcpy(v.data()+4,&z.seed,4);std::memcpy(v.data()+8,&z.octaves,4);std::memcpy(v.data()+12,&f[1],4);std::memcpy(v.data()+16,&f[2],4);int a[]={PW,z.height,z.streams,z.height/z.streams};std::memcpy(v.data()+20,a,16);return v;}
extern "C" int perlin_check(const char*,double*,double*);
