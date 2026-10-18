// Independent widened modular arithmetic oracle, does not include native RNG code.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
using U=uint64_t;using W=unsigned __int128;
U mul(U a,U b){return U(W(a)*W(b));}
U hash64(U v){U t=v^(v>>33);t=mul(t,0xff51afd7ed558ccdULL);t^=t>>33;t=mul(t,0xc4ceb9fe1a85ec53ULL);return t^(t>>33);}
uint32_t hash32(uint32_t v){U t=U(v^(v>>16))*0x85ebca6bU;uint32_t q=uint32_t(t);q^=q>>13;q=uint32_t(U(q)*0xc2b2ae35U);return q^(q>>16);}
int main(int argc,char**argv){std::string root=argv[1];const U G=1048576;std::vector<U>fixture,oracle;U sizes[3]={33,G+33,2*G+17};uint32_t seeds[3]={0,42,0xffffffffU};std::ofstream stats(root+"/oracle_statistics.json");stats<<"{\"method\":\"Independent CPU uint128 products/carry with modular64 cast, separate from source headers; selected Python arbitraryprecision controls\",\"cases\":[";
for(int ci=0;ci<3;ci++){U n=sizes[ci];uint32_t seed=seeds[ci];fixture.insert(fixture.end(),{n,U(seed),3});std::vector<U>out(n);U carries=0,zeroSeed=0;
for(U tid=0;tid<n&&tid<G;tid++){uint32_t initial=hash32(uint32_t(U(seed)+tid));U state[4]={0,0,0,initial?U(initial):4294967295ULL};zeroSeed+=initial==0;for(int round=0;round<8;round++){U value=state[3];for(int j=0;j<4;j++){value=hash64(value);state[j]=value;}}if(!state[1]||!state[2]||!state[3])return 3;
for(U index=tid;index<n;index+=G){state[0]=U(W(state[0])*6906969069ULL+1234567ULL);U a=state[1];a^=a*U(1ULL<<13);a^=a>>17;a^=a*U(1ULL<<43);state[1]=a;U old=state[2];U t=U(W(old&63ULL)*(1ULL<<58)+state[3]);W wide=W(old)+t;U carry=U(wide>>64);state[2]=U(wide);state[3]=(old/64)+carry;carries+=carry;out[index]=U(W(state[0])+state[1]+state[2]);}}
oracle.insert(oracle.end(),out.begin(),out.end());if(ci)stats<<",";stats<<"{\"n\":"<<n<<",\"seed\":"<<seed<<",\"nonzero_state_assertions_pass\":true,\"zero_local_seed_fallback_streams\":"<<zeroSeed<<",\"carry_updates\":"<<carries<<"}";}
stats<<"],\"scope\":\"Bitstream identity and state updates, not statistical quality/period certification.\"}\n";std::ofstream f(root+"/fixtures_u64.bin",std::ios::binary);f.write((char*)fixture.data(),fixture.size()*8);std::ofstream o(root+"/oracle_u64.bin",std::ios::binary);o.write((char*)oracle.data(),oracle.size()*8);printf("%zu oracle words\n",oracle.size());}
