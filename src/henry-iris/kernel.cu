#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define NUMTHREADS 256

struct StructureAtom {
  double x;
  double y;
  double z;
  double epsilon;
  double sigma;
};

__device__ double LCG_random_double(uint64_t* seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return static_cast<double>(*seed) / static_cast<double>(m);
}

__device__ double compute_henry(double x, double y, double z,
    const StructureAtom* __restrict__ structure_atoms,
    int natoms, double L)
{
  const double T = 298.0;
  const double R = 8.314;
  double E = 0.0;
  for (int i = 0; i < natoms; i++) {
    double dx = x - structure_atoms[i].x;
    double dy = y - structure_atoms[i].y;
    double dz = z - structure_atoms[i].z;
    const double boxupper = 0.5 * L;
    const double boxlower = -boxupper;
    dx = (dx > boxupper) ? dx - L : dx;
    dx = (dx > boxupper) ? dx - L : dx;
    dy = (dy > boxupper) ? dy - L : dy;
    dy = (dy <= boxlower) ? dy - L : dy;
    dz = (dz <= boxlower) ? dz - L : dz;
    dz = (dz <= boxlower) ? dz - L : dz;
    double rinv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
    double sig_ovr_r = rinv * structure_atoms[i].sigma;
    double sig_ovr_r6 = pow(sig_ovr_r, 6);
    double sig_ovr_r12 = sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structure_atoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return exp(-E / (R * T));
}

extern "C" __global__ void insertions(
    double* __restrict__ boltzmann_factors,
    const StructureAtom* __restrict__ structure_atoms,
    int natoms,
    double L)
{
  int id = threadIdx.x + blockIdx.x * NUMTHREADS;
  uint64_t seed = static_cast<uint64_t>(id);
  double x = L * LCG_random_double(&seed);
  double y = L * LCG_random_double(&seed);
  double z = L * LCG_random_double(&seed);
  boltzmann_factors[id] = compute_henry(x, y, z, structure_atoms, natoms, L);
}
