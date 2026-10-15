// Parameters.  These may change from run-to-run
#define M         (500)         // vertical dimension of the temperature array
#define RA        (100000000.f) // Rayleigh number
#define XF        (2)           // Aspect ratio for the temperature array
#define STARTSTEP (0)           // First step

// Constants.  Stability cannot be assured if these values are altered.
//#define N (XF*(M-1)+1) // horizontal dimension of temperature array
#define N          (1000)
#define DX         (1.f/(M - 1.f)) // x and z-dimensional mesh spacing
#define DX2        (DX*DX) // DX^2, for saving a few computations.
#define OMEGACOEFF (-((DX*DX)*(DX*DX))*RA) // Used on every timestep

#define PI       3.1415926535897932384626433832795028841968 // Precision here is arbitrary and may be traded
                  // for performance if context allows
#define DT_START 0.000000000000005 //timestep width.  Needs to be small to
           //guide the model through high Ra runs.


#define FRAMESIZE (DX2/4.f) // How many iterations between saves
          // flatten a 2D grid with 1D blocks into a vector.  The functionality
          // could be extended to perform pointer arithmetic, but that's not necessary
          // in this code.
          //
          //  Invoked grid geometry
          //  (2D grid with 1D blocks)
          //
          //  |---|---|---|
          //  |1,1|1,2|1,3|
          //  |---|---|---|
          //  |2,1|2,2|2,3|  ==>  1,1 1,2 1,3 2,1 2,2 ...
          //  |---|---|---|
          //  |3,1|3,2|3,3|
          //  |---|---|---|


#define THREADID (((gridDim.x)*(blockIdx.y) + (blockIdx.x))*(blockDim.x) + (threadIdx.x))

// It is possible to alternate between 2D row-major and column-major formats by
// taking transposes.
#define TPOSE(i,j,ld) ((j)*(ld) + (i))

// Simplify calling G, since many of the arguments are assured
#define SHORTG(input, compute, save, frames) \
  G(h, input, d_Tbuff, d_DxT, d_y, d_u, d_v, d_psi, d_omega, d_dsc, \
    d_dsr, d_ei, d_dt, save, h_T, compute, frames, tstep)
