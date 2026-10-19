#include <set>
#include <array>
#include <cstring>
#include "study_support.h"
#include "../gerbil-cuda/include/gerbil/KMer.h"
template<unsigned K>bool witness(const GCase&c){using T=gerbil::KMer<K>;std::set<std::vector<uint32_t>>source,fixture;bool ok=sizeof(T)==c.ip*4;for(auto&s:c.dna){std::array<unsigned char,sizeof(T)>bytes{};for(unsigned j=0;j<K;j++){auto pos=std::string("ACGT").find(s[j]);bytes[j/4]|=pos<<(2*(3-j%4));}const unsigned char*p=bytes.data();T fwd,inv;T::set(p,fwd,inv);auto norm=fwd.getNormalized(inv);ok&=memcmp(&fwd,&norm,sizeof(T))==0;auto words=(uint32_t*)&norm;source.insert(std::vector<uint32_t>(words,words+c.ip));}for(unsigned i=0;i<c.n;i++)fixture.insert(std::vector<uint32_t>(c.keys.begin()+i*c.ip,c.keys.begin()+(i+1)*c.ip));ok&=source==fixture;printf("SOURCE_KMER_WIDTH {\"k\":%u,\"sizeof_source_KMer\":%zu,\"canonical_families\":%zu,\"actual_input_keys\":%u,\"pass\":%s}\n",K,sizeof(T),source.size(),c.n,ok?"true":"false");return ok;}
int main(){auto cs=load_cases();bool ok=true;ok&=witness<8>(cs[0]);ok&=witness<28>(cs[1]);ok&=witness<60>(cs[2]);ok&=witness<92>(cs[3]);ok&=witness<135>(cs[4]);return ok?0:1;}
