#pragma once

#define IDX(x, y, z, nx, ny, nz)    ((x + (nx))%(nx) + ((y + (ny))%(ny) + ((z + (nz))%(nz))*(ny))*(nx))
#define IDF(x, y, z, f, nx, ny, nz) ((x + (nx))%(nx) + ((y + (ny))%(ny) + ((z + (nz))%(nz))*(ny))*(nx)) + (f)*(nx)*(ny)*(nz)
#define IBAR(i,nbdir) ((((i) + (nbdir)/2 - 1)%((nbdir) - 1) + 1))

#define SET_FLAG(maybe_flag, side_of_domain)\
  if (maybe_flag >= zou_he) {\
    dir[IDX(x, y, z, nx, ny, nz)] = side_of_domain;\
  }\
  if (C_p[flags[IDX(x, y, z, nx, ny, nz)]] < C_p[maybe_flag]) { \
    flags[IDX(x, y, z, nx, ny, nz)] = maybe_flag;  \
    if(maybe_flag == type_b::bounce)bounce_val[IDX(x, y, z, nx, ny, nz)] = type_b::bounce;\
  } else if ( (maybe_flag == zou_he || flags[IDX(x, y, z, nx, ny, nz)] == zou_he || maybe_flag == zou_he_p || flags[IDX(x, y, z, nx, ny, nz)] == zou_he_p)&&flags[IDX(x, y, z, nx, ny, nz)] != bounce) { \
    flags[IDX(x, y, z, nx, ny, nz)] = corner;\
  }

enum type_b { fluid, nothing, wall_m, wall, bounce, free_slip, neumann, zou_he, zou_he_p, corner, moving_wall, equilibre};
enum dir {x, x_, y, y_, z, z_};
enum stream_type { normal, tao_stream };
enum lattice_type{ D3Q19 = 19, D3Q27 = 27};

typedef char flag_type;
typedef float u_type;

struct BoxCU {
  int x0 = 0, nx = 0, y0 = 0, ny = 0, z0 = 0, nz = 0;
};

struct lbm_u {
  u_type* u0;
  u_type* u1;
  u_type* u2;
};

typedef struct {
  lbm_u u;
  lbm_u u_star;
  lbm_u g;
  double *r;
  double *f0;
  double *f1;
  flag_type *boundary_flag;
  int *boundary_values;
  flag_type *boundary_dirs;
} lbm_vars;

// Metadata used to describe the type of outer boundary to be applied to each corresponding wall
typedef struct outer_wall {
  union {
    struct {
      int xmin, xmax, ymin, ymax, zmin, zmax;
    };
    int el[6];
  };
} outer_wall;

// The discrete velocity vectors for the D3Q19 lattice
// Warning: these constants aren't actually used in the GPU computation, which
//          are unrolled for performance reason, so modifying this array won't
//          impact the actual computation.
static const char dirs[] = {
  0, 0, 0,  //  0
  -1, 0, 0, //  1
  0,-1, 0,  //  2
  0, 0,-1,  //  3
  -1,-1, 0, //  4
  -1, 1, 0, //  5
  -1, 0,-1, //  6
  -1, 0, 1, //  7
  0,-1,-1,  //  8
  0,-1, 1,  //  9
  1, 0, 0,  // 10
  0, 1, 0,  // 11
  0, 0, 1,  // 12
  1, 1, 0,  // 13
  1,-1, 0,  // 14
  1, 0, 1,  // 15
  1, 0,-1,  // 16
  0, 1, 1,  // 17
  0, 1,-1   // 18
};



static char outer_bounds_priority[12] {
    0,  // fluid,
    1,  // nothing
    2,  // wall,
    3,  // wall_m,
    7,  // bounce,
    6,  // free_slip,
    11, // neumann,
    4,  // zou_he,
    5,  // zou_he_p,
    9,  // corner,
    10, // moving_wall,
    8,  // equilibre
};
