// CPU source-policy encoder and independently structured streaming decoder.
// Input is uninterpreted UInt32 bits, not floating arithmetic.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
using U=uint32_t;using B=uint8_t;
void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
template<class T>void bytes(std::ofstream&f,const std::vector<T>&v){f.write((const char*)v.data(),v.size()*sizeof(T));}
template<class T>void guarded(std::ofstream&f,const std::vector<T>&v){B g[16];memset(g,0xa5,16);f.write((char*)g,16);bytes(f,v);f.write((char*)g,16);}
int main(int argc,char**argv){need(argc==3,"usage dataset outdir");std::string dir=argv[2];std::ifstream s(argv[1],std::ios::binary|std::ios::ate);need(bool(s),"dataset missing");auto nb=s.tellg();need(nb==50000000,"unexpected original dataset bytes");s.seekg(0);std::vector<U>real(size_t(nb)/4);s.read((char*)real.data(),nb);need(bool(s),"dataset short read");std::vector<std::vector<U>>cases{real};
for(int n:{511,512,513}){std::vector<U>x(n);U specials[]={0,0x80000000U,0x7f800000U,0xff800000U,0x7fc00001U,0x7fa12345U,0x3f800000U};for(int i=0;i<n;i++)x[i]=(i%73<40?specials[(i/8)%7]:U(i*2654435761U));cases.push_back(x);}std::vector<U>unique(2048);for(U i=0;i<unique.size();i++)unique[i]=i*2654435761U;cases.push_back(unique);cases.emplace_back(512,0);
std::ofstream input(dir+"/study_input_u32.bin",std::ios::binary),prefix(dir+"/oracle_prefix_u8.bin",std::ios::binary),oracle(dir+"/oracle_output_u8.bin",std::ios::binary),plan(dir+"/cases.txt");size_t allin=0,allout=0;
for(auto x:cases){U original=x.size(),blocks=(original+511)/512;x.resize(blocks*512,0);bytes(input,x);guarded(prefix,x);allin+=x.size()*4+32;std::vector<U>fs(blocks+1),ds(blocks+1),fo(blocks+1),doo(blocks+1);std::vector<B>tf(blocks*64),td(blocks*2048),pf,pd;
for(U b=0;b<blocks;b++){auto*in=x.data()+512*b;U cursor=0,tokens=0;while(cursor<512){U best=0,offset=0,run=0,dist=0,look=cursor;for(U w=cursor>32?cursor-32:0;w<cursor&&look<512;w++){if(in[look]==in[w]){if(!dist)dist=look-w;run++;look++;}else{if(run>best){best=run;offset=dist;}run=dist=0;look=cursor;}}if(run>best){best=run;offset=dist;}if(best){tf[b*64+tokens/8]|=B(1U<<(tokens%8));td[b*2048+ds[b]++]=B(best);td[b*2048+ds[b]++]=B(offset);cursor+=best;}else{U word=in[cursor++];for(int k=0;k<4;k++)td[b*2048+ds[b]++]=B(word>>(8*k));}tokens++;}fs[b]=(tokens+7)/8;fo[b+1]=fo[b]+fs[b];doo[b+1]=doo[b]+ds[b];pf.insert(pf.end(),tf.begin()+b*64,tf.begin()+b*64+fs[b]);pd.insert(pd.end(),td.begin()+b*2048,td.begin()+b*2048+ds[b]);}
// Independently parse the serialized stream; enforce all bounds and exact consumption.
std::vector<U>decoded(x.size());for(U b=0;b<blocks;b++){U pos=0,token=0,at=doo[b];while(pos<512){need(token/8<fs[b],"flag underrun");bool match=(pf[fo[b]+token/8]>>(token%8))&1;token++;if(match){need(at+2<=doo[b+1],"match underrun");U length=pd[at++],distance=pd[at++];need(length>0&&distance>0&&distance<=pos&&pos+length<=512,"invalid backreference");for(U j=0;j<length;j++){decoded[b*512+pos]=decoded[b*512+pos-distance];pos++;}}else{need(at+4<=doo[b+1],"literal underrun");U value=0;for(int k=0;k<4;k++)value|=U(pd[at++])<<(8*k);decoded[b*512+pos++]=value;}}need(at==doo[b+1]&&(token+7)/8==fs[b],"stream unconsumed");}need(decoded==x,"lossless decoder mismatch");
plan<<original<<" "<<blocks<<" "<<pf.size()<<" "<<pd.size()<<"\n";guarded(oracle,decoded);guarded(oracle,fs);guarded(oracle,ds);guarded(oracle,fo);guarded(oracle,doo);guarded(oracle,tf);guarded(oracle,td);guarded(oracle,pf);guarded(oracle,pd);allout+=decoded.size()*4+4*(blocks+1)*4+tf.size()+td.size()+pf.size()+pd.size()+9*32;}
std::cout<<"prefix_bytes "<<allin<<" output_bytes "<<allout<<" cases "<<cases.size()<<"\n";need(bool(input)&&bool(prefix)&&bool(oracle)&&bool(plan),"write failure");}
