__device__
float gammafunction(unsigned int n)
{   
  if(n == 0)
    return 0.0f;
  float x = ((float)n + 0.5f) * logf((float) n) - ((float)n - 1.0f);
  return x;
}

extern "C" __global__ void epi(const unsigned int* dev_data_zeros, 
                    const unsigned int* dev_data_ones, 
                    float* dev_scores, 
                    const int num_snp, 
                    const int PP_zeros, 
                    const int PP_ones,
                    const int mask_zeros, 
                    const int mask_ones) 
{
  int i, j, tid, p, k;

  j = blockDim.x * blockIdx.x + threadIdx.x; 
  i = blockDim.y * blockIdx.y + threadIdx.y;
  tid = i * num_snp + j;

  if (j > i && i < num_snp && j < num_snp) {
    unsigned int ft[2 * 9];
    for(k = 0; k < 2 * 9; k++) ft[k] = 0;

    unsigned int t00, t01, t02, t10, t11, t12, t20, t21, t22;
    unsigned int di2, dj2;
    unsigned int* SNPi;
    unsigned int* SNPj;

    // Phenotype 0
    SNPi = (unsigned int*) &dev_data_zeros[i * 2];
    SNPj = (unsigned int*) &dev_data_zeros[j * 2];

    #pragma unroll 1
    for (p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp) {
      di2 = ~(SNPi[p] | SNPi[p + 1]);
      dj2 = ~(SNPj[p] | SNPj[p + 1]);

      t00 = SNPi[p] & SNPj[p];
      t01 = SNPi[p] & SNPj[p + 1];
      t02 = SNPi[p] & dj2;
      t10 = SNPi[p + 1] & SNPj[p];
      t11 = SNPi[p + 1] & SNPj[p + 1];
      t12 = SNPi[p + 1] & dj2;
      t20 = di2 & SNPj[p];
      t21 = di2 & SNPj[p + 1];
      t22 = di2 & dj2;

      ft[0] += __popc(t00);
      ft[1] += __popc(t01);
      ft[2] += __popc(t02);
      ft[3] += __popc(t10);
      ft[4] += __popc(t11);
      ft[5] += __popc(t12);
      ft[6] += __popc(t20);
      ft[7] += __popc(t21);
      ft[8] += __popc(t22);
    }

    // remainder
    p = 2 * PP_zeros * num_snp - 2 * num_snp;
    di2 = ~(SNPi[p] | SNPi[p + 1]);
    dj2 = ~(SNPj[p] | SNPj[p + 1]);
    di2 = di2 & mask_zeros;
    dj2 = dj2 & mask_zeros;

    t00 = SNPi[p] & SNPj[p];
    t01 = SNPi[p] & SNPj[p + 1];
    t02 = SNPi[p] & dj2;
    t10 = SNPi[p + 1] & SNPj[p];
    t11 = SNPi[p + 1] & SNPj[p + 1];
    t12 = SNPi[p + 1] & dj2;
    t20 = di2 & SNPj[p];
    t21 = di2 & SNPj[p + 1];
    t22 = di2 & dj2;

    ft[0] += __popc(t00);
    ft[1] += __popc(t01);
    ft[2] += __popc(t02);
    ft[3] += __popc(t10);
    ft[4] += __popc(t11);
    ft[5] += __popc(t12);
    ft[6] += __popc(t20);
    ft[7] += __popc(t21);
    ft[8] += __popc(t22);

    // Phenotype 1
    SNPi = (unsigned int*) &dev_data_ones[i * 2];
    SNPj = (unsigned int*) &dev_data_ones[j * 2];

    #pragma unroll 1
    for(p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp)
    {
      di2 = ~(SNPi[p] | SNPi[p + 1]);
      dj2 = ~(SNPj[p] | SNPj[p + 1]);

      t00 = SNPi[p] & SNPj[p];
      t01 = SNPi[p] & SNPj[p + 1];
      t02 = SNPi[p] & dj2;
      t10 = SNPi[p + 1] & SNPj[p];
      t11 = SNPi[p + 1] & SNPj[p + 1];
      t12 = SNPi[p + 1] & dj2;
      t20 = di2 & SNPj[p];
      t21 = di2 & SNPj[p + 1];
      t22 = di2 & dj2;

      ft[9]  += __popc(t00);
      ft[10] += __popc(t01);
      ft[11] += __popc(t02);
      ft[12] += __popc(t10);
      ft[13] += __popc(t11);
      ft[14] += __popc(t12);
      ft[15] += __popc(t20);
      ft[16] += __popc(t21);
      ft[17] += __popc(t22);
    }
    p = 2 * PP_ones * num_snp - 2 * num_snp;
    di2 = ~(SNPi[p] | SNPi[p + 1]);
    dj2 = ~(SNPj[p] | SNPj[p + 1]);
    di2 = di2 & mask_ones;
    dj2 = dj2 & mask_ones;

    t00 = SNPi[p] & SNPj[p];
    t01 = SNPi[p] & SNPj[p + 1];
    t02 = SNPi[p] & dj2;
    t10 = SNPi[p + 1] & SNPj[p];
    t11 = SNPi[p + 1] & SNPj[p + 1];
    t12 = SNPi[p + 1] & dj2;
    t20 = di2 & SNPj[p];
    t21 = di2 & SNPj[p + 1];
    t22 = di2 & dj2;

    ft[9]  += __popc(t00);
    ft[10] += __popc(t01);
    ft[11] += __popc(t02);
    ft[12] += __popc(t10);
    ft[13] += __popc(t11);
    ft[14] += __popc(t12);
    ft[15] += __popc(t20);
    ft[16] += __popc(t21);
    ft[17] += __popc(t22);

    // compute score
    float score = 0.0f;

    #pragma unroll
    for(k = 0; k < 9; k++)
      score += gammafunction(ft[k] + ft[9 + k] + 1) -
               gammafunction(ft[k]) - gammafunction(ft[9 + k]);
    score = fabsf(score);
    if(score == 0.0f)
      score = FLT_MAX;
    dev_scores[tid] = score;
  }
}
