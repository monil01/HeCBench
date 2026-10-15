#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define HECBENCH_LEUKO_VALIDATE 1
inline FILE* leuk_file(const char* name,const char* mode){
 const char* root=getenv("HECBENCH_LEUKOCYTE_DUMP");if(!root)return nullptr;
 FILE*f=fopen((std::string(root)+"/"+name).c_str(),mode);
 if(!f){perror(name);exit(2);}return f;
}
template<class T>inline void leuk_write(FILE*f,const T*p,size_t n){if(fwrite(p,sizeof(T),n,f)!=n){perror("dump");exit(2);}}
template<class T>inline void leuk_dump(const char*name,const T*p,size_t n){if(FILE*f=leuk_file(name,"wb")){leuk_write(f,p,n);if(fclose(f))exit(2);}}
inline void leuk_track(int frame,int cells,int snaxels,const double*x,const double*y,double**r){
 if(FILE*f=leuk_file("trajectory.bin","ab")){
  int shape[]={frame,cells,snaxels};leuk_write(f,shape,3);
  for(int c=0;c<cells;++c){leuk_write(f,x+c,1);leuk_write(f,y+c,1);leuk_write(f,r[c],snaxels);}
  if(fclose(f))exit(2);
 }
}
inline void leuk_mgvf(int id,int cells,int total,const int*off,const int*m,const int*n,const float*in,const float*out){
 if(FILE*f=leuk_file("mgvf.bin","ab")){
  int shape[]={id,cells,total};leuk_write(f,shape,3);
  leuk_write(f,off,cells);leuk_write(f,m,cells);leuk_write(f,n,cells);
  leuk_write(f,in,total);leuk_write(f,out,total);if(fclose(f))exit(2);
 }
}
