#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "../rng-wallace-cuda/rand_helpers.h"
#include "../rng-wallace-cuda/constants.h"
#include "../rng-wallace-cuda/reference.h"
void save(const char*f,const std::vector<float>&v){FILE*p=fopen(f,"wb");if(!p)exit(1);fwrite(v.data(),4,v.size(),p);fclose(p);}
int main(){const int nb=257,n=nb*2048;std::vector<float>pool(n+17,-12345),chi(nb+17,-12345);for(int i=0;i<n;++i)pool[i]=RandN();for(int i=0;i<nb;++i)chi[i]=MakeChi2Scale(n);std::vector<float>fixture={257,float(n),3,1,0,1,22136,4660,7,65535,65535,31};fixture.insert(fixture.end(),pool.begin(),pool.end());fixture.insert(fixture.end(),chi.begin(),chi.end());save("fixture.bin",fixture);std::vector<float>gold;unsigned seeds[]={1,0x12345678u,0xffffffffu};int steps[]={1,7,31};for(int c=0;c<3;++c){auto p=pool;std::vector<float>out(n+17,-12345);for(int k=0;k<steps[c];++k)reference(seeds[c],p.data(),out.data(),chi.data(),nb);gold.insert(gold.end(),p.begin(),p.end());gold.insert(gold.end(),out.begin(),out.end());}save("source_cpu_outputs.bin",gold);}
