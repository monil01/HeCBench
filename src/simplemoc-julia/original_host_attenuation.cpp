#include <cuda.h>
#include "../simplemoc-cuda/SimpleMOC-kernel_header.h"

// host 
void attenuate_segment(
    Input *I, Source *S,
    int QSR_id, int FAI_id, float *state_flux,
    SIMD_Vectors *simd_vecs)
{
  // Unload local vector vectors
  float *q0 =            simd_vecs->q0;
  float *q1 =            simd_vecs->q1;
  float *q2 =            simd_vecs->q2;
  float *sigT =          simd_vecs->sigT;
  float *tau =           simd_vecs->tau;
  float *sigT2 =         simd_vecs->sigT2;
  float *expVal =        simd_vecs->expVal;
  float *reuse =         simd_vecs->reuse;
  float *flux_integral = simd_vecs->flux_integral;
  float *tally =         simd_vecs->tally;
  float *t1 =            simd_vecs->t1;
  float *t2 =            simd_vecs->t2;
  float *t3 =            simd_vecs->t3;
  float *t4 =            simd_vecs->t4;

  // Some placeholder constants - In the full app some of these are
  // calculated based off position in geometry. This treatment
  // shaves off a few FLOPS, but is not significant compared to the
  // rest of the function.
  const float dz = 0.1f;
  const float zin = 0.3f; 
  const float weight = 0.5f;
  const float mu = 0.9f;
  const float mu2 = 0.3f;
  const float ds = 0.7f;

  const int egroups = I->egroups;

  // load fine source region flux vector
  float * FSR_flux = &S[QSR_id].fine_flux[FAI_id * egroups];

  if( FAI_id == 0 )
  {
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
    float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups]; 
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y2 = f2[g];
      const float y3 = f3[g];

      // do linear "fitting"
      const float c0 = y2;
      const float c1 = (y3 - y2) / dz;

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin;
      q1[g] = c1;
      q2[g] = 0;
    }
  }
  else if ( FAI_id == I->fine_axial_intervals - 1 )
  {
    float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups]; 
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y1 = f1[g];
      const float y2 = f2[g];

      // do linear "fitting"
      const float c0 = y2;
      const float c1 = (y2 - y1) / dz;

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin;
      q1[g] = c1;
      q2[g] = 0;
    }
  }
  else
  {
    float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups]; 
    float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
    float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups]; 
    // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
    for( int g = 0; g < egroups; g++)
    {
      // load neighboring sources
      const float y1 = f1[g]; 
      const float y2 = f2[g];
      const float y3 = f3[g];

      // do quadratic "fitting"
      const float c0 = y2;
      const float c1 = (y1 - y3) / (2.f*dz);
      const float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

      // calculate q0, q1, q2
      q0[g] = c0 + c1*zin + c2*zin*zin;
      q1[g] = c1 + 2.f*c2*zin;
      q2[g] = c2;
    }
  }


  // cycle over energy groups
#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // load total cross section
    sigT[g] = S[QSR_id].sigT[g];

    // calculate common values for efficiency
    tau[g] = sigT[g] * ds;
    sigT2[g] = sigT[g] * sigT[g];
  }


  // cycle over energy groups
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    expVal[g] = 1.f - expf( -tau[g] ); // exp is faster on many architectures
  }

  // Flux Integral

  // Re-used Term
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g] 
      / (sigT[g] * sigT2[g]); 
  }


  //#pragma vector alignednontemporal
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // add contribution to new source flux
    flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * state_flux[g] - q0[g])
        * expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2 
      * (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g]) 
      / (3.f * sigT2[g] * sigT2[g]);
  }


#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    // Prepare tally
    tally[g] = weight * flux_integral[g];
  }

#ifdef INTEL
#pragma vector
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    FSR_flux[g] += tally[g];
  }


  // Term 1
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t1[g] = q0[g] * expVal[g] / sigT[g];  
  }
  // Term 2
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g]; 
  }
  // Term 3
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t3[g] =  q2[g] * mu2 * reuse[g];
  }
  // Term 4
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    t4[g] = state_flux[g] * (1.f - expVal[g]);
  }

#ifdef VERIFY
  for( int g = 0; g < egroups; g++) {
    printf("q0[%d] = %f\n", g, q0[g]);
    printf("q1[%d] = %f\n", g, q1[g]);
    printf("q2[%d] = %f\n", g, q2[g]);
    printf("sigT[%d] = %f\n", g, sigT[g]);
    printf("tau[%d] = %f\n", g, tau[g]);
    printf("sigT2[%d] = %f\n", g, sigT2[g]);
    printf("expVal[%d] = %f\n", g, expVal[g]);
    printf("reuse[%d] = %f\n", g, reuse[g]);
    printf("flux_integral[%d] = %f\n", g, flux_integral[g]);
    printf("tally[%d] = %f\n", g, tally[g]);
    printf("t1[%d] = %f\n", g, t1[g]);
    printf("t2[%d] = %f\n", g, t2[g]);
    printf("t3[%d] = %f\n", g, t3[g]);
    printf("t4[%d] = %f\n", g, t4[g]);
  }
#endif

  // Total psi
#ifdef INTEL
#pragma vector aligned
#elif defined IBM
#pragma vector_level(10)
#endif
  for( int g = 0; g < egroups; g++)
  {
    state_flux[g] = t1[g] + t2[g] + t3[g] + t4[g];
  }
}  

