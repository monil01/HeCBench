#define main preserved_native_main
#include "../mr-cuda/main.cu"
#undef main
#include <vector>
int main(){for(int mode=0;mode<2;mode++){if(mode==0)set_nprimes();else set_nintegers();const char*name=mode?"study_odd_u32.bin":"study_prime_u32.bin";FILE*f=fopen(name,"wb");fwrite(n32,sizeof(uint32_t),4*BENCHMARK_ITERATIONS,f);fclose(f);printf("%s values=%d source_rng_exact=1\n",name,4*BENCHMARK_ITERATIONS);}return 0;}
