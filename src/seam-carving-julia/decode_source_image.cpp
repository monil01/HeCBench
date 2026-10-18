#include <cstdio>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "../seam-carving-cuda/stb_image.h"
int main(){int w,h,n;auto p=stbi_load("../seam-carving-cuda/Broadway_tower.jpg",&w,&h,&n,0);if(!p||w!=128||h!=128||n!=3)return 1;FILE*f=fopen("study_source_rgb_u8.bin","wb");fwrite(p,1,w*h*3,f);fclose(f);printf("{\"source_STB_decoded\":true,\"w\":%d,\"h\":%d,\"components\":%d}\n",w,h,n);free(p);}
