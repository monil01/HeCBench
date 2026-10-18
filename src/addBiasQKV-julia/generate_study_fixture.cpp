#include "../addBiasQKV-cuda/reference.h"
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstdio>
template<class T>void save(std::string p,const std::vector<T>&a){std::ofstream f(p,std::ios::binary);f.write((char*)a.data(),a.size()*sizeof(T));}
int main(int argc,char**argv){std::string dir=argv[1];for(int c=0;c<2;c++){int m=c?5:8192,h=c?3:12,d=c?16:64,n=m*3*h*d,b=3*h*d;std::string tag=c?"axis":"original";std::vector<uint8_t>s(n),out(n);std::vector<uint16_t>bias(b);std::mt19937 rng(123);std::uniform_real_distribution<float>dist(-2,2);for(int j=0;j<n;j++){int k=j%254;s[j]=c?uint8_t(k<127?k:k+1):ref::f32_to_fp8_e4m3(dist(rng),false);}for(int j=0;j<b;j++)bias[j]=ref::f32_to_bf16(c?float(j%97-48)*.03125f:dist(rng));std::vector<float>sc=c?std::vector<float>{.5013f,1.2491f}:std::vector<float>{.5f,1.25f};addBiasQKV_reference(out.data(),s.data(),bias.data(),sc[0],sc[1],m,h,d,h*d,false);save(dir+"/study_"+tag+"_src_u8.bin",s);save(dir+"/study_"+tag+"_bias_u16.bin",bias);save(dir+"/study_"+tag+"_scales_f32.bin",sc);save(dir+"/study_"+tag+"_expected_u8.bin",out);printf("%s m%d h%d d%d count%d\n",tag.c_str(),m,h,d,n);}}
