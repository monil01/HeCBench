#include <random>
#include <algorithm>
#include <vector>
#include <climits>
#include <cstdio>
#include <cstdint>
template<class T>void write(const char*n,const std::vector<T>&a){FILE*f=fopen(n,"wb");if(!f)exit(3);fwrite(a.data(),sizeof(T),a.size(),f);fclose(f);}
template<class T>void generate(bool fp){constexpr size_t N=1048576;std::mt19937 rng(19937);std::vector<T>keys(N);if(fp){std::uniform_real_distribution<float>u(0,1);for(auto&x:keys)x=T(u(rng));}else{std::uniform_int_distribution<unsigned>u(0,UINT_MAX);for(auto&x:keys)x=T(u(rng));}write(fp?"study_input_f32.bin":"study_input_u32.bin",keys);std::vector<std::pair<T,unsigned>>pairs;for(unsigned i=0;i<N;i++)pairs.push_back({keys[i],i});std::stable_sort(pairs.begin(),pairs.end(),[](auto a,auto b){return a.first<b.first;});std::vector<unsigned>values(N);size_t ties=0;for(size_t i=0;i<N;i++){keys[i]=pairs[i].first;values[i]=pairs[i].second;if(i&&keys[i]==keys[i-1])ties++;}write(fp?"study_expected_keys_f32.bin":"study_expected_keys_u32.bin",keys);write(fp?"study_expected_values_f32_u32.bin":"study_expected_values_u32.bin",values);printf("CPU_STABLE_SORT dtype=%s N=%zu duplicateadjacencies=%zu\n",fp?"float32":"uint32",N,ties);}
int main(){generate<float>(true);generate<unsigned>(false);std::vector<unsigned>values(1048576);for(unsigned i=0;i<values.size();i++)values[i]=i;write("study_input_values_u32.bin",values);}
