static float study_sigma=2;
float study_rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  float V1 = sqrtf(-2 * logf(U1)) * cosf(2 * PI * U2);
  return V1;
}

/*
   Generate random gamma variables on a CPU.
 */
float study_rgamma(float a, float b)
{
  float x;
  if (a > 1) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    bool flag = true;
    float V;
    while (flag) {
      // Generate a standard normal random variable
      float Z = study_rnorm();
      if (Z > -1.f / c) {
        V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
      }
    }
    x = d * V / b;
  }
  else 
  {
    x = study_rgamma(a + 1.f, b);
    x = x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
  return x;
}

/*
   Metropolis algorithm for producing random a values. 
   The proposal distribution in normal with a variance that
   is adjusted at each step.
 */
float study_sample_a(float a, float b, int N, float log_sum)
{
  float &sigma = study_sigma;

  float proposal = study_rnorm() * sigma + a;

  if(proposal <= 0) return a;

  float log_acceptance_ratio = (proposal - a) * log_sum +
    N * (proposal - a) * logf(b) -
    N * (lgamma(proposal) - lgamma(a));

  float U = rand() / float(RAND_MAX);

  if(logf(U) < log_acceptance_ratio){
    sigma *= 1.1;
    return proposal;
  } else {
    sigma /= 1.1;
    return a;
  }
}

/*
   Returns a random b sample (simply a draw from the appropriate
   gamma distribution)
 */
float study_sample_b(float a, int N, float flat_sum)
{
  float hyperA = N * a + 1;
  float hyperB = flat_sum;
  return study_rgamma(hyperA, hyperB);
}

