#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
inline void diagnostic_dump_results(resultsStruct results,int n) {
 const char* path=getenv("HECBENCH_DUMP");if(!path) return;
 FILE* f=fopen(path,"wb");if(!f) throw std::runtime_error("cannot open results dump");
 uint64_t count=n;
 if(fwrite(&count,sizeof(count),1,f)!=1) throw std::runtime_error("count dump");
 dataType* arrays[]={results.dirtyPrice,results.accruedAmountCurrDate,results.cleanPrice,results.bondForwardVal};
 for(auto a:arrays) if(fwrite(a,sizeof(dataType),n,f)!=size_t(n)) throw std::runtime_error("results dump");
 fclose(f);
}
