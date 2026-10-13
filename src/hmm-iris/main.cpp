#include "iris_dmem.hpp"
#include <utility>
#ifndef HMM_STATES
#define HMM_STATES 4096
#endif
#ifndef HMM_EMISSIONS
#define HMM_EMISSIONS 4096
#endif
#ifndef HMM_OBSERVATIONS
#define HMM_OBSERVATIONS 250
#endif
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <stdio.h>

// forward declaractions
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit);
int ViterbiCPU(float &viterbiProb,
               int *viterbiPath,
               int *obs, 
               const int &nObs, 
               float *initProb,
               float *mtState, 
               const int &nState,
               float *mtEmit);
int ViterbiGPU(float &viterbiProb,
               int *__restrict__ viterbiPath,
               int *__restrict__ obs, 
               const int nObs, 
               float *__restrict__ initProb,
               float *__restrict__ mtState, 
               const int nState,
               const int nEmit,
               float *__restrict__ mtEmit);



// main function
//*****************************************************************************
int main(int argc, char **argv)
{
    int nState = HMM_STATES; // number of states, must be a multiple of 256
    int nEmit  = HMM_EMISSIONS; // number of possible observations
    int nDevice = 1;
    
    float *initProb = (float*)malloc(sizeof(float)*nState); // initial probability
    float *mtState  = (float*)malloc(sizeof(float)*nState*nState); // state transition matrix
    float *mtEmit   = (float*)malloc(sizeof(float)*nEmit*nState); // emission matrix
    initHMM(initProb, mtState, mtEmit, nState, nEmit);

    // define observational sequence
    int nObs = HMM_OBSERVATIONS; // size of observational sequence
    int **obs = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathCPU = (int**)malloc(nDevice*sizeof(int*));
    int **viterbiPathGPU = (int**)malloc(nDevice*sizeof(int*));
    float *viterbiProbCPU = (float*)malloc(nDevice*sizeof(float)); 
    float *viterbiProbGPU = (float*)malloc(nDevice*sizeof(float)); 
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        obs[iDevice] = (int*)malloc(sizeof(int)*nObs);
        for (int i = 0; i < nObs; i++)
            obs[iDevice][i] = i % 15;
        viterbiPathCPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
        viterbiPathGPU[iDevice] = (int*)malloc(sizeof(int)*nObs);
    }

    printf("# of states = %d\n# of possible observations = %d \nSize of observational sequence = %d\n\n",
        nState, nEmit, nObs);



    IRISSession session(&argc,&argv);
    printf("\nCompute Viterbi path on GPU\n");
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ViterbiGPU(viterbiProbGPU[iDevice], viterbiPathGPU[iDevice], obs[iDevice], nObs, initProb, mtState, nState, nEmit, mtEmit);
    }

    printf("\nCompute Viterbi path on CPU\n");
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        ViterbiCPU(viterbiProbCPU[iDevice], viterbiPathCPU[iDevice], obs[iDevice], nObs, initProb, mtState, nState, mtEmit);
    }
    
    bool pass = true;
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        for (int i = 0; i < nObs; i++)
        {
            if (viterbiPathCPU[iDevice][i] != viterbiPathGPU[iDevice][i]) 
            {
                pass = false;
                break;
            }
        }
    }

    if (pass)
      printf("Success");
    else
      printf("Fail");
    printf("\n");
        
    free(initProb);
    free(mtState);
    free(mtEmit);
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        free(obs[iDevice]);
        free(viterbiPathCPU[iDevice]);
        free(viterbiPathGPU[iDevice]);
    }
    free(obs);
    free(viterbiPathCPU);
    free(viterbiPathGPU);
    free(viterbiProbCPU);
    free(viterbiProbGPU);

    return 0;

}

// initialize initial probability, state transition matrix and emission matrix with random 
// numbers. Note that this does not satisfy the normalization property of the state matrix. 
// However, since the algorithm does not use this property, for testing purpose this is fine.
//*****************************************************************************
int initHMM(float *initProb, float *mtState, float *mtObs, const int &nState, const int &nEmit)
{
    if (nState <= 0 || nEmit <=0) return 0;

    // Initialize initial probability

    for (int i = 0; i < nState; i++) initProb[i] = rand();
    float sum = 0.0;
    for (int i = 0; i < nState; i++) sum += initProb[i];
    for (int i = 0; i < nState; i++) initProb[i] /= sum;

    // Initialize state transition matrix

    for (int i = 0; i < nState; i++) {
        for (int j = 0; j < nState; j++) {
            mtState[i*nState + j] = rand();
            mtState[i*nState + j] /= RAND_MAX;
        }
    }

    // init emission matrix

    for (int i = 0; i < nEmit; i++)
    {
        for (int j = 0; j < nState; j++) 
        {
            mtObs[i*nState + j] = rand();
        }
    }

    // normalize the emission matrix
    for (int j = 0; j < nState; j++) 
    {
        float sum = 0.0;
        for (int i = 0; i < nEmit; i++) sum += mtObs[i*nState + j];
        for (int i = 0; i < nEmit; i++) mtObs[i*nState + j] /= sum;
    }

    return 1;
}
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "iris_dmem.hpp"

// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
int ViterbiGPU(float &viterbiProb,
    int   *viterbiPath,
    int   *obs, 
    const int nObs, 
    float *initProb,
    float *mtState, 
    const int nState,
    const int nEmit,
    float *mtEmit)
{

  float maxProbNew[nState];
  int path[(nObs-1)*nState];

  float *d_mtState, *d_mtEmit, *d_maxProbOld, *d_maxProbNew;
  int *d_obs, *d_path;

  iris_alloc((void**)&d_mtState, sizeof(float)*nState*nState);
  iris_alloc((void**)&d_mtEmit, sizeof(float)*nEmit*nState);
  iris_alloc((void**)&d_obs, sizeof(int)*nObs);
  iris_alloc((void**)&d_maxProbOld, sizeof(float)*nState);
  iris_alloc((void**)&d_maxProbNew, sizeof(float)*nState);
  iris_alloc((void**)&d_path, sizeof(int)*(nObs-1)*nState);

  iris_copy(d_mtState, mtState, sizeof(float)*nState*nState, IRISHostToDevice);
  iris_copy(d_mtEmit, mtEmit, sizeof(float)*nEmit*nState, IRISHostToDevice);
  iris_copy(d_obs, obs, sizeof(int)*nObs, IRISHostToDevice);

  dim3 gridDim ((nState + 255)/256);
  dim3 blockDim (256);

  // initial probability
  iris_copy(d_maxProbOld, initProb, sizeof(float)*nState, IRISHostToDevice);

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  { 
    iris_launch("viterbi", gridDim, blockDim, d_maxProbOld, d_mtState, d_mtEmit, d_obs, d_maxProbNew, d_path, nState, t);
    std::swap(d_maxProbOld, d_maxProbNew);
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time of Viterbi iterations %f (s)\n", time * 1e-9f);

  iris_copy(maxProbNew, d_maxProbOld, sizeof(float)*nState, IRISDeviceToHost);
  iris_copy(path, d_path, sizeof(int)*(nObs-1)*nState, IRISDeviceToHost);

  // find the final most probable state
  float maxProb = 0.0;
  int maxState = -1;
  for (int i = 0; i < nState; i++) 
  {
    if (maxProbNew[i] > maxProb) 
    {
      maxProb = maxProbNew[i];
      maxState = i;
    }
  }
  viterbiProb = maxProb;

  // backtrace to find the Viterbi path
  viterbiPath[nObs-1] = maxState;
  for (int t = nObs-2; t >= 0; t--) 
  {
    viterbiPath[t] = path[t*nState+viterbiPath[t+1]];
  }

  iris_free(d_mtState);
  iris_free(d_mtEmit);
  iris_free(d_obs);
  iris_free(d_maxProbOld);
  iris_free(d_maxProbNew);
  iris_free(d_path);

  return 1;
}
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#include <cstdlib>
#include <cstdio>

///////////////////////////////////////////////////////////////////////////////
// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
///////////////////////////////////////////////////////////////////////////////
int ViterbiCPU(float &viterbiProb,
               int *viterbiPath,
               int *obs, 
               const int &nObs, 
               float *initProb,
               float *mtState, 
               const int &nState,
               float *mtEmit)
{
    float *maxProbNew = (float*)malloc(sizeof(float)*nState);
    float *maxProbOld = (float*)malloc(sizeof(float)*nState);
    int **path = (int**)malloc(sizeof(int*)*(nObs-1));
    for (int i = 0; i < nObs-1; i++)
        path[i] = (int*)malloc(sizeof(int)*nState);

    // initial probability
    #pragma omp parallel for
    for (int i = 0; i < nState; i++)
    {
        maxProbOld[i] = initProb[i];
    }

    // main iteration of Viterbi algorithm
    for (int t = 1; t < nObs; t++) // for every input observation
    { 
        #pragma omp parallel for
        for (int iState = 0; iState < nState; iState++) 
        {
            // find the most probable previous state leading to iState
            float maxProb = 0.0;
            int maxState = -1;
            for (int preState = 0; preState < nState; preState++) 
            {
                float p = maxProbOld[preState] + mtState[iState*nState + preState];
                if (p > maxProb) 
                {
                    maxProb = p;
                    maxState = preState;
                }
            }
            maxProbNew[iState] = maxProb + mtEmit[obs[t]*nState+iState];
            path[t-1][iState] = maxState;
        }
        
        #pragma omp parallel for
        for (int iState = 0; iState < nState; iState++) 
        {
            maxProbOld[iState] = maxProbNew[iState];
        }
    }
    
    // find the final most probable state
    float maxProb = 0.0;
    int maxState = -1;
    for (int i = 0; i < nState; i++) 
    {
        if (maxProbNew[i] > maxProb) 
        {
            maxProb = maxProbNew[i];
            maxState = i;
        }
    }
    viterbiProb = maxProb;

    // backtrace to find the Viterbi path
    viterbiPath[nObs-1] = maxState;
    for (int t = nObs-2; t >= 0; t--) 
    {
        viterbiPath[t] = path[t][viterbiPath[t+1]];
    }

    free(maxProbNew);
    free(maxProbOld);
    for (int i = 0; i < nObs-1; i++) free(path[i]);
    free(path);
    return 1;
}
