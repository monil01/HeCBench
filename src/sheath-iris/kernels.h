__device__ double XtoL(double pos);
__device__ double gather(double lc, const double* field);
__device__ void scatter(double lc, float value, float* field);
extern "C" __global__ void scatterParticle(const Particle* __restrict particles, float*__restrict den, long N)
{
  /*get particle id*/
  long p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < N && particles[p].alive)
  {
    double lc = XtoL(particles[p].x);
    scatter(lc, 1.f, den);
  }
}
extern "C" __global__ void pushParticle(Particle*__restrict particles, const double*__restrict ef, double qm, long N)
{
  /*get particle id*/
  long p = blockIdx.x * blockDim.x + threadIdx.x;

  if (p < N && particles[p].alive)
  {
    /*grab pointer to this particle*/
    Particle* part = &particles[p];

    /*compute particle node position*/
    double lc = XtoL(part->x);

    /*gather electric field onto particle position*/
    double part_ef = gather(lc, ef);

    /*advance velocity*/
    part->v += DT * qm * part_ef;

    /*advance position*/
    part->x += DT * part->v;

    /*remove particles leaving the domain*/
    if (part->x < X0 || part->x >= XMAX)
      part->alive = false;
  }
}
extern "C" __global__ void rewindParticle(Particle*__restrict particles, const double*__restrict ef, double qm, long N)
{
  /*get particle id*/
  long p = blockIdx.x * blockDim.x + threadIdx.x;

  if (p < N && particles[p].alive)
  {
    /*grab pointer to this particle*/
    Particle* part = &particles[p];

    /*compute particle node position*/
    double lc = XtoL(part->x);

    /*gather electric field onto particle position*/
    double part_ef = gather(lc, ef);

    /*advance velocity*/
    part->v -= 0.5 * DT * qm * part_ef;
  }
}
__device__ double XtoL(double pos)
{
  double li = (pos - 0) / DX;
  return li;
}
__device__ void scatter(double lc, float value, float* field)
{
  int i    = (int)lc;
  float di = lc - i;
  atomicAdd(&(field[i]), value * (1 - di));
  atomicAdd(&(field[i + 1]), value * (di));
}
__device__ double gather(double lc, const double* field)
{
  int i     = (int)lc;
  double di = lc - i;

  /*gather field value onto particle position*/
  double val = field[i] * (1 - di) + field[i + 1] * (di);
  return val;
}
