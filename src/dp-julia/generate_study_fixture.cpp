#include <random>
#include <vector>
#include <fstream>
#include <cstdint>
int main(){std::ofstream f("study_inputs_i32.bin",std::ios::binary);for(int n:{4099,16387}){int pad=((n+1023)/1024)*1024;std::vector<int32_t>a(pad),b(pad);std::mt19937 e(19937);std::uniform_int_distribution<int>d(-32,32);for(int i=0;i<n;i++){a[i]=d(e);b[i]=d(e);}int32_t meta[]={n,pad};f.write((char*)meta,8);f.write((char*)a.data(),pad*4);f.write((char*)b.data(),pad*4);}return f?0:1;}
