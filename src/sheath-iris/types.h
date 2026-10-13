/*constants*/
#define EPS_0 8.85418782e-12 // F/m, vacuum permittivity
#define K 1.38065e-23        // J/K, Boltzmann constant
#define ME 9.10938215e-31    // kg, electron mass
#define QE 1.602176565e-19   // C, elementary charge
#define AMU 1.660538921e-27  // kg, atomic mass unit
#define EV_TO_K 11604.52     // 1eV in Kelvin, QE/K

/*simulation parameters, these could come from an input file*/
#define PLASMA_DEN 1e16      // plasma density to load
#ifndef HECBENCH_PARTICLES
#define HECBENCH_PARTICLES 500000
#endif
#define NUM_IONS HECBENCH_PARTICLES      // number of ions
#define NUM_ELECTRONS HECBENCH_PARTICLES // number of electrons
#define DX 1e-4              // cell spacing
#define NC 100               // number of cells
#ifndef HECBENCH_STEPS
#define HECBENCH_STEPS 1000
#endif
#define NUM_TS HECBENCH_STEPS          // number of time steps
#define DT 1e-11             // time step size
#define ELECTRON_TEMP 3.0    // electron temperature in eV
#define ION_TEMP 1.0         // ion temperature in eV

/*domain parameters, set here so can access from GPU*/
#define X0 0           /*mesh origin*/
#define XL NC* DX      /*domain length*/
#define XMAX (X0 + XL) /*domain max position*/

const int THREADS_PER_BLOCK = 256;
/* Data structure to hold domain information*/
struct Domain
{
  const int ni      = NC + 1; /*number of nodes*/
  const double x0   = X0;
  const double dx   = DX;
  const double xl   = XL;
  const double xmax = XMAX;

  /*data structures*/
  double* phi; /*potential*/
  double* ef;  /*electric field on the cpu*/
  double* rho; /*charge density*/

  float* ndi; /*ion density on the CPU*/
  float* nde; /*electron density on the CPU*/
};

/* Data structure for particle storage **/
struct Particle
{
  double x;   /*position*/
  double v;   /*velocity*/
  bool alive; /*flag to avoid removing particles*/
};

/* Data structure to hold species information*/
struct Species
{
  double mass;   /*particle mass in kg*/
  double charge; /*particle charge in Coulomb*/
  double spwt;   /*species specific weight*/

  int np;             /*number of particles*/
  int np_alloc;       /*size of the allocated data array*/
  Particle* part;     /*array holding particles on the CPU*/
};
