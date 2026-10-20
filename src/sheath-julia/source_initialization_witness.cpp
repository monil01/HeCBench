#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <iostream>
/*constants*/
#define EPS_0 8.85418782e-12 // F/m, vacuum permittivity
#define K 1.38065e-23        // J/K, Boltzmann constant
#define ME 9.10938215e-31    // kg, electron mass
#define QE 1.602176565e-19   // C, elementary charge
#define AMU 1.660538921e-27  // kg, atomic mass unit
#define EV_TO_K 11604.52     // 1eV in Kelvin, QE/K

/*simulation parameters, these could come from an input file*/
#define PLASMA_DEN 1e16      // plasma density to load
#define NUM_IONS 500000      // number of ions
#define NUM_ELECTRONS 500000 // number of electrons
#define DX 1e-4              // cell spacing
#define NC 100               // number of cells
#define NUM_TS 1000          // number of time steps
#define DT 1e-11             // time step size
#define ELECTRON_TEMP 3.0    // electron temperature in eV
#define ION_TEMP 1.0         // ion temperature in eV

/*domain parameters, set here so can access from GPU*/
#define X0 0           /*mesh origin*/
#define XL NC* DX      /*domain length*/
#define XMAX (X0 + XL) /*domain max position*/


double rnd()
{
  return rand() / (double)RAND_MAX;
}
double SampleVel(double v_th)
{
  const int M = 12;
  double sum  = 0;
  for (int i = 0; i < M; i++)
    sum += rnd();

  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);int hdr[3];f.read((char*)hdr,12);if(hdr[0]!=NUM_IONS||hdr[1]!=1||hdr[2]!=0)return 3;srand(123);unsigned long checked=0;for(int species=0;species<2;species++){int n=species?NUM_ELECTRONS:NUM_IONS;double mass=species?ME:16*AMU;double temp=species?ELECTRON_TEMP:ION_TEMP;double delta=(NC*DX)/n;double vth=sqrt(2*K*temp*EV_TO_K/mass);for(int i=0;i<n;i++){double wantx=X0+i*delta,wantv=SampleVel(vth),x,v;f.read((char*)&x,8);f.read((char*)&v,8);if(!f||memcmp(&x,&wantx,8)||memcmp(&v,&wantv,8))return 4;checked+=2;}}if(f.peek()!=EOF)return 5;std::cout<<"SourceExactInitializedDoubleValues "<<checked<<" PASS\n";}
