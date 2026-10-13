// https://www.particleincell.com/2016/cuda-pic/
// https://www.particleincell.com/wp-content/uploads/2016/02/sheath-gpu.cu


/* 1D sheath PIC simulation with CUDA */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "dmem_runtime.hpp"

#define CUDA_ERROR(err) iris_check(err)
#include "types.h"
#include "dispatch.h"
/** FUNCTION PROTOTYPES **/
double rnd();
double SampleVel(double v_th);
void ScatterSpecies(Species* species, Particle* species_part_gpu, float* den, float* den_gpu, double &time);
void ComputeRho(Species* ions, Species* electrons);
bool SolvePotential(double* phi, double* rho);
bool SolvePotentialDirect(double* phi, double* rho);
void ComputeEF(double* phi, double* ef, double* ef_gpu);
void PushSpecies(Species* species, Particle* species_part_gpu, double* ef);
void RewindSpecies(Species* species, Particle* species_part_gpu, double* ef);
void AddParticle(Species* species, double x, double v);
__device__ double XtoL(double pos);
__device__ double gather(double lc, const double* field);
__device__ void scatter(double lc, float value, float* field);

void WriteResults(int ts);

/* GLOBAL VARIABLES */
Domain domain;

FILE* file_res;

/* --------- main -------------*/
int main(int argc, char* argv[])
{
  int p;
  int ts; // time step
  double sp_time = 0.0; // total time of the scatter-particle kernel

  domain.phi = new double[domain.ni]; // potential
  domain.rho = new double[domain.ni]; // charge density
  domain.ef  = new double[domain.ni]; // electric field
  domain.nde = new float[domain.ni];  // number density of electrons
  domain.ndi = new float[domain.ni];  // number density of ions

  /*save pointers so we can write phi instead of domain.phi*/
  double* phi = domain.phi;
  double* rho = domain.rho;
  double* ef  = domain.ef;
  float* nde  = domain.nde;
  float* ndi  = domain.ndi;

  /*also allocate device memory */
  float *nde_gpu, *ndi_gpu;
  CUDA_ERROR(iris_alloc((void**)&nde_gpu, domain.ni * sizeof(float)));
  CUDA_ERROR(iris_alloc((void**)&ndi_gpu, domain.ni * sizeof(float)));

  double* ef_gpu;
  CUDA_ERROR(iris_alloc((void**)&ef_gpu, domain.ni * sizeof(double)));

  Particle *ions_part_gpu;
  CUDA_ERROR(iris_alloc((void**)&ions_part_gpu, NUM_IONS * sizeof(Particle)));

  Particle *electrons_part_gpu;
  CUDA_ERROR(iris_alloc((void**)&electrons_part_gpu, NUM_ELECTRONS * sizeof(Particle)));

  /*clear data*/
  memset(phi, 0, sizeof(double) * domain.ni);

  /*variables to hold species data ions*/
  Species ions;
  Species electrons;

  /*set material data*/
  ions.mass     = 16 * AMU;
  ions.charge   = QE;
  ions.spwt     = PLASMA_DEN * domain.xl / NUM_IONS;
  ions.np       = 0;
  ions.np_alloc = NUM_IONS;
  ions.part     = new Particle[NUM_IONS];

  electrons.mass     = ME; // electrons
  electrons.charge   = -QE;
  electrons.spwt     = PLASMA_DEN * domain.xl / NUM_ELECTRONS;
  electrons.np       = 0;
  electrons.np_alloc = NUM_ELECTRONS;
  electrons.part     = new Particle[NUM_ELECTRONS];

  /*randomize RNG*/
  srand(123);

  /*load uniformly spaced ions */
  double delta_ions = domain.xl / NUM_IONS;
  double v_thi      = sqrt(2 * K * ION_TEMP * EV_TO_K / ions.mass);
  for (p = 0; p < NUM_IONS; p++)
  {
    double x = domain.x0 + p * delta_ions;
    double v = SampleVel(v_thi);
    AddParticle(&ions, x, v);
  }

  /*load uniformly spaced electrons */
  double delta_electrons = domain.xl / NUM_ELECTRONS;
  double v_the           = sqrt(2 * K * ELECTRON_TEMP * EV_TO_K / electrons.mass);
  for (p = 0; p < NUM_ELECTRONS; p++)
  {
    double x = domain.x0 + p * delta_electrons;
    double v = SampleVel(v_the);
    AddParticle(&electrons, x, v);
  }

  /*copy particles to the GPU*/
  CUDA_ERROR(iris_copy(ions_part_gpu, ions.part,
    NUM_IONS * sizeof(Particle), iris_h2d));

  CUDA_ERROR(iris_copy(electrons_part_gpu, electrons.part, 
    NUM_ELECTRONS * sizeof(Particle), iris_h2d));

  /*compute number density*/
  ScatterSpecies(&ions, ions_part_gpu, ndi, ndi_gpu, sp_time);
  ScatterSpecies(&electrons, electrons_part_gpu, nde, nde_gpu, sp_time);

  /*compute charge density and solve potential*/
  ComputeRho(&ions, &electrons);

  SolvePotential(phi, rho);

  ComputeEF(phi, ef, ef_gpu);

  RewindSpecies(&ions, ions_part_gpu, ef_gpu);
  RewindSpecies(&electrons, electrons_part_gpu, ef_gpu);

  /*OUTPUT*/
  file_res = fopen("result.dat", "w");
  fprintf(file_res, "VARIABLES = x nde ndi rho phi ef\n");
  WriteResults(0);

  auto start = std::chrono::steady_clock::now();

  /* MAIN LOOP*/
  for (ts = 1; ts <= NUM_TS; ts++)
  {
    /*compute number density*/
    ScatterSpecies(&ions, ions_part_gpu, ndi, ndi_gpu, sp_time);
    ScatterSpecies(&electrons, electrons_part_gpu, nde, nde_gpu, sp_time);

    ComputeRho(&ions, &electrons);
    SolvePotential(phi, rho);
    ComputeEF(phi, ef, ef_gpu);

    /*move particles*/
    PushSpecies(&electrons, electrons_part_gpu, ef_gpu);
    PushSpecies(&ions, ions_part_gpu, ef_gpu);

    /*write diagnostics*/
    if (ts % 25 == 0)
    {
      /*max phi*/
      double max_phi = abs(phi[0]);
      for (int i = 0; i < domain.ni; i++)
        if (abs(phi[i]) > max_phi)
          max_phi = abs(phi[i]);

      printf("TS:%i\tnp_i:%d\tnp_e:%d\tdphi:%.3g\n", ts, ions.np, electrons.np,
          max_phi - phi[0]);
    }

    /*save data*/
    if (ts % 1000 == 0) WriteResults(ts);
  }
  iris_sync();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  if (NUM_TS % 1000 != 0) WriteResults(NUM_TS);
  fclose(file_res);

  /*free up memory*/
  delete phi;
  delete rho;
  delete ef;
  delete nde;
  delete ndi;
  iris_free(nde_gpu);
  iris_free(ndi_gpu);
  iris_free(ef_gpu);

  /*free particles*/
  delete ions.part;
  delete electrons.part;
  iris_free(ions_part_gpu);
  iris_free(electrons_part_gpu);

  printf("Total kernel execution time (scatter particles) : %.3g (s)\n", sp_time * 1e-9f),
  printf("Total time for %d time steps: %.3g (s)\n", NUM_TS, time * 1e-9f);
  printf("Time per time step: %.3g (ms)\n", (time * 1e-6f) / NUM_TS);

  /* call cudaDeviceReset for correct profiling data*/
  iris_finish();

  iris_finish();
  return 0;
}

/***** HELPER FUNCTIONS *********************************************************/
/* random number generator
   for now using built-in but this is not adequate for real simulations*/
double rnd()
{
  return rand() / (double)RAND_MAX;
}

/* samples random velocity from Maxwellian distribution using Birdsall's method*/
double SampleVel(double v_th)
{
  const int M = 12;
  double sum  = 0;
  for (int i = 0; i < M; i++)
    sum += rnd();

  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}


/* move particles*/


/*scatter particles of species to the mesh*/
void ScatterSpecies(Species* species, Particle* species_part_gpu,
                    float* den, float* den_gpu, double &time)
{
  /*initialize densities to zero*/
  CUDA_ERROR(iris_fill(den_gpu, 0, sizeof(float) * domain.ni));

  int size = species->np_alloc;

  /*scatter particles to the mesh*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  dispatch_scatterParticle( dim3(nblocks), dim3(THREADS_PER_BLOCK), species_part_gpu, den_gpu, size);

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  /*copy density back to CPU*/
  CUDA_ERROR(iris_copy(den, den_gpu, sizeof(float) * domain.ni, iris_d2h));

  /*divide by cell volume*/
  for (int i = 0; i < domain.ni; i++)
    den[i] *= species->spwt / domain.dx;

  /*only half cell at boundaries*/
  den[0] *= 2.0;
  den[domain.ni - 1] *= 2.0;
}

/*adds new particle to the species, returns pointer to the newly added data*/
void AddParticle(Species* species, double x, double v)
{
  /*abort the simulation if we ran out of space to store this particle*/
  if (species->np > species->np_alloc - 1)
  {
    printf("Too many particles!\n");
    exit(-1);
  }

  /*store position and velocity of this particle*/
  species->part[species->np].x     = x;
  species->part[species->np].v     = v;
  species->part[species->np].alive = true;

  /*increment particle counter*/
  species->np++;
}

/*computes charge density by adding ion and electron data*/
void ComputeRho(Species* ions, Species* electrons)
{
  double* rho = domain.rho;

  for (int i = 0; i < domain.ni; i++)
    rho[i] = ions->charge * domain.ndi[i] + electrons->charge * domain.nde[i];
}

/*Thomas algorithm for a tri-diagonal matrix*/
bool SolvePotentialDirect(double* x, double* rho)
{
  /*set coefficients, this should be pre-computed*/
  int ni     = domain.ni;
  double dx2 = domain.dx * domain.dx;
  int i;
  double* a = new double[ni];
  double* b = new double[ni];
  double* c = new double[ni];

  /*central difference on internal nodes*/
  for (i = 1; i < ni - 1; i++)
  {
    a[i] = 1;
    b[i] = -2;
    c[i] = 1;
  }

  /*dirichlet b.c. on boundaries*/
  a[0]      = 0;
  b[0]      = 1;
  c[0]      = 0;
  a[ni - 1] = 0;
  b[ni - 1] = 1;
  c[ni - 1] = 0;

  /*multiply RHS*/
  for (i = 1; i < domain.ni - 1; i++)
    x[i] = -rho[i] * dx2 / EPS_0;

  x[0]      = 0;
  x[ni - 1] = 0;

  /* Modify the coefficients. */
  c[0] /= b[0]; /* Division by zero risk. */
  x[0] /= b[0]; /* Division by zero would imply a singular matrix. */
  for (i = 1; i < ni; i++)
  {
    double id = (b[i] - c[i - 1] * a[i]); /* Division by zero risk. */
    c[i] /= id;                           /* Last value calculated is redundant. */
    x[i] = (x[i] - x[i - 1] * a[i]) / id;
  }

  /* Now back substitute. */
  for (i = ni - 2; i >= 0; i--)
    x[i] = x[i] - c[i] * x[i + 1];

  return true;
}

/* solves potential using the Gauss Seidel Method, returns true if converged*/
bool SolvePotential(double* phi, double* rho)
{
  double L2;
  double dx2 = domain.dx * domain.dx; /*precompute*/

  /*initialize boundaries*/
  phi[0] = phi[domain.ni - 1] = 0;

  /*solve potential, identical to lesson 2*/
  for (int solver_it = 0; solver_it < 40000; solver_it++)
  {
    /*Gauss Seidel method, phi[i-1]-2*phi[i]+phi[i+1] = -dx^2*rho[i]/eps_0*/
    for (int i = 1; i < domain.ni - 1; i++)
    {
      /*SOR*/
      double g = 0.5 * (phi[i - 1] + phi[i + 1] + dx2 * rho[i] / EPS_0);
      phi[i]   = phi[i] + 1.4 * (g - phi[i]);
    }

    /*check for convergence*/
    if (solver_it % 25 == 0)
    {
      double sum = 0;
      for (int i = 1; i < domain.ni - 1; i++)
      {
        double R = -rho[i] / EPS_0 - (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / dx2;
        sum += R * R;
      }
      L2 = sqrt(sum) / domain.ni;
      if (L2 < 1e-4)
      {
        return true;
      }
    }
  }
  printf("Gauss-Seidel solver failed to converge, L2=%.3g!\n", L2);
  return false;
}

/* computes electric field by differentiating potential*/
void ComputeEF(double* phi, double* ef, double* ef_gpu)
{
  for (int i = 1; i < domain.ni - 1; i++)
    ef[i] = -(phi[i + 1] - phi[i - 1]) / (2 * domain.dx); // central difference

  /*one sided difference at boundaries*/
  ef[0]             = -(phi[1] - phi[0]) / domain.dx;
  ef[domain.ni - 1] = -(phi[domain.ni - 1] - phi[domain.ni - 2]) / domain.dx;

  /*copy to the gpu*/
  CUDA_ERROR(iris_copy(ef_gpu, ef, domain.ni * sizeof(double), iris_h2d));
}

/*GPU code to move particles*/


/* moves particles of a single species, returns wall charge*/
void PushSpecies(Species* species, Particle* species_part_gpu, double* ef)
{
  /*precompute q/m*/
  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  /*loop over particles*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;
  dispatch_pushParticle( dim3(nblocks), dim3(THREADS_PER_BLOCK), species_part_gpu, ef, qm, size);
}



/*GPU code to rewind particles*/



/* rewinds particle velocities by -0.5DT*/
void RewindSpecies(Species* species, Particle* species_part_gpu, double* ef)
{
  /*precompute q/m*/
  double qm = species->charge / species->mass;

  int size = species->np_alloc;

  /*loop over particles*/
  int nblocks = 1 + size / THREADS_PER_BLOCK;
  dispatch_rewindParticle( dim3(nblocks), dim3(THREADS_PER_BLOCK), species_part_gpu, ef, qm, size);
}


/* converts physical coordinate to logical*/



/* atomic scatter of scalar value onto a field at logical coordinate lc*/


/* gathers field value at logical coordinate lc*/



/* writes new zone to the results file*/
void WriteResults(int ts)
{
  fprintf(file_res, "ZONE I=%d T=ZONE_%06d\n", domain.ni, ts);
  for (int i = 0; i < domain.ni; i++)
  {
    fprintf(file_res, "%g %g %g %g %g %g\n", i * domain.dx, domain.nde[i], domain.ndi[i],
        domain.rho[i], domain.phi[i], domain.ef[i]);
  }

  fflush(file_res);
}
