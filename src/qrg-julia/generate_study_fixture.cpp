
#include "../qrg-cuda/reference.cu"
#include <cstdio>
#include <cstdint>
int main(){unsigned int table[93];initQuasirandomGenerator(table);double prefix[112];prefix[0]=65537;prefix[1]=0;prefix[2]=196611;prefix[3]=0xffffffffU/196612U;prefix[4]=20;for(int i=0;i<93;i++)prefix[5+i]=table[i];unsigned int probes[]={0,1,2,0x147ae146,0x147ae147,0x147ae148,0x7ffffffe,0x7fffffff,0x80000000,0x80000001,0xeb851eb7,0xfffffffd,0xfffffffe,0xffffffff};for(int i=0;i<14;i++)prefix[98+i]=probes[i];FILE*f=fopen("study_input_f64.bin","wb");fwrite(prefix,8,112,f);fclose(f);f=fopen("study_cjn_i64.bin","wb");for(int dim=0;dim<3;dim++)for(int b=0;b<63;b++)fwrite(&cjn[b][dim],8,1,f);fclose(f);}
