#define BSIZE 16
#define XTILE 20

typedef double Real;

extern "C" __global__ void stencil3d(
    const Real* __restrict__ d_psi,
    Real* __restrict__ d_npsi,
    const Real* __restrict__ d_sigmaX,
    const Real* __restrict__ d_sigmaY,
    const Real* __restrict__ d_sigmaZ,
    int nx, int ny, int nz)
{
  __shared__ Real sm_psi[4][BSIZE][BSIZE];

#define V0(y, z) sm_psi[pii][y][z]
#define V1(y, z) sm_psi[cii][y][z]
#define V2(y, z) sm_psi[nii][y][z]
#define sigmaX(x, y, z, dir) d_sigmaX[(z) + nz * ((y) + ny * ((x) + nx * (dir)))]
#define sigmaY(x, y, z, dir) d_sigmaY[(z) + nz * ((y) + ny * ((x) + nx * (dir)))]
#define sigmaZ(x, y, z, dir) d_sigmaZ[(z) + nz * ((y) + ny * ((x) + nx * (dir)))]
#define psi(x, y, z) d_psi[(z) + nz * ((y) + ny * (x))]
#define npsi(x, y, z) d_npsi[(z) + nz * ((y) + ny * (x))]

  const int tjj = threadIdx.y;
  const int tkk = threadIdx.x;

  d_psi = &(psi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z));
  d_npsi = &(npsi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z));
  d_sigmaX = &(sigmaX(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z, 0));
  d_sigmaY = &(sigmaY(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z, 0));
  d_sigmaZ = &(sigmaZ(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z, 0));

  int nLast_x = XTILE + 1;
  int nLast_y = BSIZE - 1;
  int nLast_z = BSIZE - 1;
  if (blockIdx.x == gridDim.x - 1) nLast_x = nx - 2 - XTILE * blockIdx.x + 1;
  if (blockIdx.y == gridDim.y - 1) nLast_y = ny - 2 - (BSIZE - 2) * blockIdx.y + 1;
  if (blockIdx.z == gridDim.z - 1) nLast_z = nz - 2 - (BSIZE - 2) * blockIdx.z + 1;
  if (tjj > nLast_y || tkk > nLast_z) return;

  int pii = 0, cii = 1, nii = 2, tii;
  sm_psi[cii][tjj][tkk] = psi(0, tjj, tkk);
  sm_psi[nii][tjj][tkk] = psi(1, tjj, tkk);
  Real xcharge, ycharge, zcharge, dV = 0;
  __syncthreads();

  if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
    Real xd = -V1(tjj, tkk) + V2(tjj, tkk);
    Real yd = (-V1(-1 + tjj, tkk) + V1(1 + tjj, tkk) - V2(-1 + tjj, tkk) + V2(1 + tjj, tkk)) / 4.;
    Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V2(tjj, -1 + tkk) + V2(tjj, 1 + tkk)) / 4.;
    dV -= sigmaX(1, tjj, tkk, 0) * xd + sigmaX(1, tjj, tkk, 1) * yd + sigmaX(1, tjj, tkk, 2) * zd;
  }

  tii = pii; pii = cii; cii = nii; nii = tii;
  for (int ii = 1; ii < nLast_x; ii++) {
    sm_psi[nii][tjj][tkk] = psi(ii + 1, tjj, tkk);
    __syncthreads();
    if ((tkk > 0) && (tkk < nLast_z) && (tjj < nLast_y)) {
      Real xd = (-V0(tjj, tkk) - V0(1 + tjj, tkk) + V2(tjj, tkk) + V2(1 + tjj, tkk)) / 4.;
      Real yd = -V1(tjj, tkk) + V1(1 + tjj, tkk);
      Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V1(1 + tjj, -1 + tkk) + V1(1 + tjj, 1 + tkk)) / 4.;
      ycharge = sigmaY(ii, tjj + 1, tkk, 0) * xd + sigmaY(ii, tjj + 1, tkk, 1) * yd + sigmaY(ii, tjj + 1, tkk, 2) * zd;
      dV += ycharge;
      sm_psi[3][tjj][tkk] = ycharge;
    }
    __syncthreads();
    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) dV -= sm_psi[3][tjj - 1][tkk];
    __syncthreads();
    if ((tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
      Real xd = (-V0(tjj, tkk) - V0(tjj, 1 + tkk) + V2(tjj, tkk) + V2(tjj, 1 + tkk)) / 4.;
      Real yd = (-V1(-1 + tjj, tkk) - V1(-1 + tjj, 1 + tkk) + V1(1 + tjj, tkk) + V1(1 + tjj, 1 + tkk)) / 4.;
      Real zd = -V1(tjj, tkk) + V1(tjj, 1 + tkk);
      zcharge = sigmaZ(ii, tjj, tkk + 1, 0) * xd + sigmaZ(ii, tjj, tkk + 1, 1) * yd + sigmaZ(ii, tjj, tkk + 1, 2) * zd;
      dV += zcharge;
      sm_psi[3][tjj][tkk] = zcharge;
    }
    __syncthreads();
    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) dV -= sm_psi[3][tjj][tkk - 1];
    __syncthreads();
    if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
      Real xd = -V1(tjj, tkk) + V2(tjj, tkk);
      Real yd = (-V1(-1 + tjj, tkk) + V1(1 + tjj, tkk) - V2(-1 + tjj, tkk) + V2(1 + tjj, tkk)) / 4.;
      Real zd = (-V1(tjj, -1 + tkk) + V1(tjj, 1 + tkk) - V2(tjj, -1 + tkk) + V2(tjj, 1 + tkk)) / 4.;
      xcharge = sigmaX(ii + 1, tjj, tkk, 0) * xd + sigmaX(ii + 1, tjj, tkk, 1) * yd + sigmaX(ii + 1, tjj, tkk, 2) * zd;
      dV += xcharge;
      npsi(ii, tjj, tkk) = dV;
      dV = -xcharge;
    }
    __syncthreads();
    tii = pii; pii = cii; cii = nii; nii = tii;
  }
}
