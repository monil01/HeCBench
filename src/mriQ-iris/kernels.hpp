extern "C" __global__ void
ComputePhiMag_GPU(
  const float* __restrict__ phiR,
  const float* __restrict__ phiI,
        float* __restrict__ phiMag,
  const int numK)
{
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

extern "C" __global__ void
ComputeQ_GPU(
  const int numK,
        int kGlobalIndex,
  const float* __restrict__ x,
  const float* __restrict__ y,
  const float* __restrict__ z,
        float* __restrict__ Qr,
        float* __restrict__ Qi,
  const kValues* __restrict__ values,
  const int numX)
{
  // Determine the element of the X arrays computed by this thread
  int xIndex = blockIdx.x*KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

  if(xIndex>=numX) return;
  const kValues* ck=values+kGlobalIndex;
  const int tile_count = numK-kGlobalIndex < KERNEL_Q_K_ELEMS_PER_GRID ?
      numK-kGlobalIndex : KERNEL_Q_K_ELEMS_PER_GRID;
  // Read block's X values from global mem to shared mem
  float sX = x[xIndex];
  float sY = y[xIndex];
  float sZ = z[xIndex];
  float sQr = Qr[xIndex];
  float sQi = Qi[xIndex];

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  if (tile_count % 2) {
    float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
    sQr += ck[0].PhiMag * cosf(expArg);
    sQi += ck[0].PhiMag * sinf(expArg);
    kIndex++;
    kGlobalIndex++;
  }

  for (; (kIndex < tile_count) && (kGlobalIndex < numK);
       kIndex += 2, kGlobalIndex += 2) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr += ck[kIndex].PhiMag * cosf(expArg);
    sQi += ck[kIndex].PhiMag * sinf(expArg);

    int kIndex1 = kIndex + 1;
    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX +
			    ck[kIndex1].Ky * sY +
			    ck[kIndex1].Kz * sZ);
    sQr += ck[kIndex1].PhiMag * cosf(expArg1);
    sQi += ck[kIndex1].PhiMag * sinf(expArg1);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}
