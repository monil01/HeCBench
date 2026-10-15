#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
// Opt-in complete trace: every warmup and measured generation, in call order.
inline void mrg_trace(const std::vector<float>& data,const char* kind,bool reset=false) {
  const char* prefix=std::getenv("HECBENCH_MRG_DUMP"); if(!prefix) return;
  std::string path=std::string(prefix)+"."+kind+".bin";
  FILE* file=std::fopen(path.c_str(),reset?"wb":"ab");
  if(!file) { std::perror(path.c_str()); std::exit(2); }
  bool ok=reset || std::fwrite(data.data(),sizeof(float),data.size(),file)==data.size();
  if(std::fclose(file) || !ok) std::exit(2);
}
