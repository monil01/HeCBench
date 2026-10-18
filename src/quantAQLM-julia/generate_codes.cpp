#include <cstdio>
#include <cstdlib>
#include <cstdint>
int main(int argc,char**argv){if(argc!=2)return 2;FILE*f=fopen(argv[1],"wb");if(!f)return 3;srand(123);for(int i=0;i<12288*512;i++){int16_t x=rand()%65536-32768;fwrite(&x,2,1,f);}return fclose(f);}
