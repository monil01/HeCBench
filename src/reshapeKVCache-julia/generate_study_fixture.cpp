#include <vector>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstdint>
#include <cmath>
template<class T>void write(const char*n,const std::vector<T>&a){FILE*f=fopen(n,"wb");if(!f)exit(3);if(fwrite(a.data(),sizeof(T),a.size(),f)!=a.size())exit(3);fclose(f);}
int main(){constexpr int N=1024,H=32,D=128;constexpr size_t Q=size_t(N)*H*D;std::vector<float>key(Q),value(Q);std::vector<int64_t>slot(N);std::mt19937 gen(42);std::normal_distribution<float>dist(0,1);float km=0,vm=0;for(size_t i=0;i<Q;i++){key[i]=dist(gen);value[i]=dist(gen);km=std::max(km,key[i]);vm=std::max(vm,value[i]);if(!std::isfinite(key[i])||!std::isfinite(value[i]))return 4;}for(int i=0;i<N;i++)slot[i]=i;std::shuffle(slot.begin(),slot.end(),gen);write("study_key_f32.bin",key);write("study_value_f32.bin",value);write("study_slot_i64.bin",slot);write("study_scales_f32.bin",std::vector<float>{km/64.f,vm/64.f});printf("SOURCE_FIXTURE N=%d H=%d D=%d seed42 kmax=%g vmax=%g kscale=%g vscale=%g values=%zu\n",N,H,D,km,vm,km/64.f,vm/64.f,Q);}
