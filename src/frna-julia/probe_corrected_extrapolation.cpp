// Original function and original parameter preparation; no GPU or source correction.
#define INT
#include "configured_native/frna.cu"
#include <string>
int main(int argc,char**argv){if(argc!=2)return 2;fparam p{};fparam_read_from_text(argv[1],0,&p);printf("{\"conversion_factor\":%.9g,\"prelog\":%.9g,\"parsed_extrapolation_Int32\":%d,\"hairpin_init30\":%d,\"terminal_mismatch_CAAG\":%d,\"cases\":[",double(conversion_factor),double(p.prelog),p.Extrapolation_for_large_loops,p.hairpin_loop_initiation[30],p.tstackh[C][G][A][A]);int ds[]={30,40,60};for(int k=0;k<3;k++){int d=ds[k];std::string str="C"+std::string(d,'A')+"G";fbase_t*s=fsequence_from_string(str.c_str());int e=hairpin_loop_energy(s,0,d+1,d,&p);printf("%s{\"unpaired_bases\":%d,\"energy_Int32\":%d}",k?",":"",d,e);free(s);}printf("]}\n");}
