#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "SDKBitMap.h"
int main(){SDKBitMap bitmap;bitmap.load("../urng-sycl/URNG_Input.bmp");if(!bitmap.isLoaded())return 1;size_t n=bitmap.getHeight()*bitmap.getWidth();printf("%d %d %zu\n",bitmap.getWidth(),bitmap.getHeight(),n);std::vector<uchar4>out(bitmap.getPixels(),bitmap.getPixels()+n);for(int k=0;k<1024;k++){unsigned char v=k%256;out.push_back(make_uchar4(v,v,v,v));}uchar4 tail[7]={make_uchar4(0,255,0,255),make_uchar4(255,0,255,0),make_uchar4(1,2,3,4),make_uchar4(254,255,253,252),make_uchar4(0,0,0,0),make_uchar4(255,255,255,255),make_uchar4(17,63,129,241)};out.insert(out.end(),tail,tail+7);FILE*f=fopen("fixture_rgba.bin","wb");fwrite(out.data(),4,out.size(),f);fclose(f);}
