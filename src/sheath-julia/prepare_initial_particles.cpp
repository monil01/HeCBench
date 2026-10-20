// Exact original initialized species values; named smaller cases disclosed.
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
static constexpr double K=1.38065e-23,EV=11604.52,AMU=1.660538921e-27,ME=9.10938215e-31,XL=100*1e-4;
static double velocity(double vth){double sum=0;for(int j=0;j<12;j++)sum+=rand()/(double)RAND_MAX;return sqrt(.5)*vth*(sum-6.)/sqrt(12./12.);}
int main(int argc,char**argv){if(argc!=2)return 2;int ns[]={500000,4099,257},steps[]={1,25,1000};for(int c=0;c<3;c++){int32_t n=ns[c],ts=steps[c],cold=c==2;std::ofstream f(std::string(argv[1])+"/case"+std::to_string(c)+".bin",std::ios::binary);f.write((char*)&n,4);f.write((char*)&ts,4);f.write((char*)&cold,4);srand(123);double dx=XL/n;for(int s=0;s<2;s++){double vth=sqrt(2*K*(s?3.:1.)*EV/(s?ME:16*AMU));for(int i=0;i<n;i++){double x=i*dx,v=velocity(vth);if(cold){x=(i%100+.5)*1e-4;v=0.;}f.write((char*)&x,8);f.write((char*)&v,8);}}if(!f)return 3;std::cout<<"case="<<c<<" n="<<n<<" timesteps="<<ts<<" coldneutral="<<cold<<"\n";}}
