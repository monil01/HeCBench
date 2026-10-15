#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
inline void projection_dump(int batch,const char*field,const void*data,size_t bytes){
 const char*directory=std::getenv("DAPHNE_DUMP");if(!directory)return;
 std::string name=std::string(directory)+"/batch_"+std::to_string(batch)+"_"+field+".bin";
 FILE*f=std::fopen(name.c_str(),"wb");if(!f){std::perror(name.c_str());std::exit(1);}
 if(std::fwrite(data,1,bytes,f)!=bytes){std::fclose(f);std::exit(1);}if(std::fclose(f))std::exit(1);
}
