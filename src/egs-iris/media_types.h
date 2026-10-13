#pragma once
#define MXGE        200
#define MXRAYFF     100
#define RAYCDFSIZE  100
#define MXELEMENT   100
#define MXSHELL     6

typedef struct element_t {
  char    symbol[2];
  double  z;
  double  wa;
  double  pz;
  double  rhoz;
} element_t;

typedef struct medium_t {
  string  *name;
  double  rho;
  double  rlc;
  double  ae, ap;
  double  ue, up;
  uint    ne;
  element_t *elements;
  int     iunrst, epstfl, iaprim;
} medium_t;

typedef struct __align__(16) rayleigh_data_t {
  float   xgrid;
  float   fcum;
  float   b_array;
  float   c_array;
} rayleigh_data_t;

double binding_energies[MXSHELL][MXELEMENT];

float2          *d_ge, *d_gmfp, *d_gbr1, *d_gbr2, *d_cohe, *d_pmax;
rayleigh_data_t *d_rayleigh_data;
uint            *d_i_array;

#ifdef IRIS_DEVICE
__device__ float2             *ge, *gmfp, *gbr1, *gbr2, *cohe, *pmax;
#endif
#ifdef IRIS_DEVICE
__device__ rayleigh_data_t    *rayleigh_data;
#endif
#ifdef IRIS_DEVICE
__device__ uint               *i_array;
#endif
