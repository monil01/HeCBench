// Chiranjit Mukherjee  (chiranjit@soe.ucsc.edu)

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

#define SSS     // Runs the Stochastic Shotgun Search
// #define USE_GPU // If uncommented, runs kernels on GPU
#define MAXL 10 // Maximum number of mixture components that can be accommodated

#ifdef SSS
// SSS- runtime parameters
// #define maxLocalWastedIterations (n+p)  // In paper, C
#define maxLocalWastedIterations 50

#define climbDownStepSize 10               // In paper, D

// #define maxLocalJumpCount 10            // In paper, R
#define maxLocalJumpCount 3

#define MAXNGLOBALJUMP 2                   // In paper, S / (C * R)
// SSS- parameters for lists of models saved

#define sizeOfFeatureSelectionList 20      // In paper, M

#define sizeOfBestList 100 // Number of highest-score models to keep track of
// SSS-
#define LOCALMOVE_SFACTOR 0.001
#define GLOBALJUMP_SFACTOR 0.01
#define G_TO_XI int(L * p * (p - 1) / (2 * n)) // In paper, g
#define XI_TO_SM 10                            // In paper, h
#define LOOKFORWARD 5                          // In paper, f
#define RGMS_T 2                               // In paper, t
// number of chains parameters
#define N_INIT                                                                 \
  1 // Number of points of initial models provided by the user in folder DATA/

#define TRY_EACH_INIT                                                          \
  1 // Number of times to restart from each given initial point

#define N_RANDOM_RESTART                                                       \
  1 // Number of times to restart from random random initial points

#define N_MODES_LIST_RESTART 1 // Number of times to start from

// #define maxNmodes
// ((TRY_EACH_INIT*N_INIT+N_RANDOM_RESTART+N_MODES_LIST_RESTART)+1)
#define maxNmodes 2
#endif

#define PI 3.1415926
#define log_2 0.693147180559945
#define log_pi_over_4 0.286182471462350
#define log_2_pi 1.837877066409345
#define NEG_INF -999999.0
#define myBool bool
#define myInt short // Using short interger
#define myIntFactor 2
#define intFactor 4
// #define Real double
#define Real float // Using floating-point
#define ISFLOAT 10
using namespace std;

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#define GSL_INTEGRATION_GRIDSIZE 1000
gsl_integration_workspace *w;
gsl_function F;

#include <gsl/gsl_randist.h>
#define RANDOMSEED 314159265

// Define hyperparameters for the prior distribution of (mu, K | G)
#define N0 0.01
#define DELTA0 3
#define JEFFREYS_PRIOR
gsl_rng *rnd;

#ifdef USE_GPU
#include <cuda.h>
#define BLOCK_SIZE 32
#define SYNC __syncthreads()
typedef struct {
  cudaStream_t delete_stream;
  cudaStream_t add_stream;
  myInt *d_in_delete;
  myInt *d_in_add;
  myInt *d_which_delete;
  myInt *d_which_add;
  myInt *h_in_delete;
  myInt *h_in_add;
  myInt *which_delete;
  myInt *which_add;
  int n_add, n_delete;
} MGPUstuff;
#else
typedef struct {
} MGPUstuff;
#endif
MGPUstuff device;


#include "original_native/graph.cpp"
#include <vector>
#include <cstdint>
#include <stdexcept>
static bool chordal(const std::vector<uint64_t>&a){int n=a.size();uint64_t used=0;std::vector<int>w(n);for(int step=0;step<n;step++){int v=-1;for(int j=0;j<n;j++)if(!(used>>j&1)&&(v<0||w[j]>w[v]))v=j;uint64_t s=a[v]&used;for(int j=0;j<n;j++)if(s>>j&1)if((s&~(uint64_t(1)<<j))&~a[j])return false;used|=uint64_t(1)<<v;for(int j=0;j<n;j++)if(!(used>>j&1)&&(a[v]>>j&1))w[j]++;}return true;}
static bool noinducedcycle(const std::vector<uint64_t>&a){int n=a.size();for(uint64_t s=0;s<(uint64_t(1)<<n);s++){if(__builtin_popcountll(s)<4)continue;bool two=true;for(int j=0;j<n;j++)if((s>>j&1)&&__builtin_popcountll(a[j]&s)!=2)two=false;if(!two)continue;uint64_t reached=s&-s,old;do{old=reached;for(int j=0;j<n;j++)if(reached>>j&1)reached|=a[j]&s;}while(old!=reached);if(reached==s)return false;}return true;}
int main(){uint64_t graphs=0,valid=0,sourceclassificationerrors=0,gateerrors=0,gatechecks=0;const int n=6;for(uint64_t bits=0;bits<(uint64_t(1)<<15);bits++){std::vector<uint64_t>a(n);int q=0;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++,q++)if(bits>>q&1){a[i]|=uint64_t(1)<<j;a[j]|=uint64_t(1)<<i;}bool ok=chordal(a);if(ok!=noinducedcycle(a))throw std::runtime_error("independentMCSvsinducedcyclewitness");Graph g;g.InitGraph(n);for(int i=0;i<n;i++)for(int j=0;j<n;j++)g.Edge[i][j]=(a[i]>>j)&1;bool src=g.IsDecomposable();graphs++;if(src!=ok){sourceclassificationerrors++;if(sourceclassificationerrors<5)printf("SSS_CPU_CLASS_WITNESS bits=%zu source=%d oracle=%d\n",bits,src,ok);}if(!ok||!src)continue;valid++;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){auto b=a;b[i]^=uint64_t(1)<<j;b[j]^=uint64_t(1)<<i;bool expected=chordal(b);bool actual=(a[i]>>j&1)?(g.CanDeleteEdge(i,j)>=0):g.CanAddEdge(i,j);gatechecks++;if(actual!=expected){gateerrors++;if(gateerrors<5)printf("SSS_CPU_GATE_WITNESS bits=%zu i=%d j=%d present=%d source=%d oracle=%d\n",bits,i,j,int(a[i]>>j&1),actual,expected);}}}
printf("SSS_CPU_EXHAUSTIVE graphs=%zu chordal=%zu gatechecks=%zu classification_errors=%zu gate_errors=%zu\n",graphs,valid,gatechecks,sourceclassificationerrors,gateerrors);return(sourceclassificationerrors||gateerrors)?2:0;}
