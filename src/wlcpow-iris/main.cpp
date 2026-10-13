// Modified for IRIS DMem/HIP on 2026-10-13; GNU GPL v3, see LICENSE.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include "iris_support.h"
#include <cstring>
#include "vector_types.h"

template <typename T>
T* resize (int n) {
  return (T*) malloc (sizeof(T) * n);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  if(repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();

  int i;

  // all the values are randomly initialized

  float3 period = {0.5f, 0.5f, 0.5f};
  int padding = 1;
  int n_type = 32;
  int n = 1e6;  // problem size

  float4 *coord_merged = resize<float4>(n+1);
  float4 *veloc = resize<float4>(n+1);
  int *nbond = resize<int>(n+1);

  // set the sizes properly
  int2 *bonds = resize<int2>(n+n+1);
  r64 *bond_r0 = resize<r64>(n+n+1);

  r64 *force_x = resize<r64>(n+1);
  r64 *force_y = resize<r64>(n+1);
  r64 *force_z = resize<r64>(n+1);

  r32 *bond_l0 = resize<r32>(n+1);
  r32 *temp = resize<r32>(n+1);
  r32 *mu_targ = resize<r32>(n+1);
  r32 *qp = resize<r32>(n+1);
  r32 *gamc = resize<r32>(n+1);
  r32 *gamt = resize<r32>(n+1);
  r32 *sigc = resize<r32>(n+1);
  r32 *sigt = resize<r32>(n+1);

  std::mt19937 g (19937);
  std::uniform_real_distribution<r64> dist_r64(0.1, 0.9);
  std::uniform_real_distribution<r32> dist_r32(0.1, 0.9);
  std::uniform_int_distribution<i32> dist_i32(0, n_type);

  for (i = 0; i < n + n + 1; i++) {
    bond_r0[i] = dist_r64(g) + 0.001;
    // select two distinct atoms in the kernel to evaluate their forces
    bonds[i] = { (i+1)%(n+1), 
                 dist_i32(g) };
  }

  for (i = 0; i < n + 1; i++) {
    nbond[i] = dist_i32(g);
    coord_merged[i] = {dist_r32(g), dist_r32(g), dist_r32(g), 0};
    r32 vx = dist_r32(g), vy = dist_r32(g), vz = dist_r32(g);
    veloc[i] = {vx, vy, vz, sqrtf(vx*vx+vy*vy+vz*vz)};
    bond_l0[i] = dist_r32(g);
    gamt[i] = dist_r32(g);
    gamc[i] = ((dist_i32(g) % 4) + 4) * gamt[i]; // gamt[i] <= 3.0*gamc[i]
    temp[i] = dist_r32(g);
    mu_targ[i] = dist_r32(g);
    qp[i] = dist_r32(g);
    sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]));
    sigt[i] = 2.0*sqrt(gamt[i]*temp[i]);
  }

  memset(force_x,0,sizeof(r64)*(n+1));
  memset(force_y,0,sizeof(r64)*(n+1));
  memset(force_z,0,sizeof(r64)*(n+1));
  iris::DMem dev_coord_merged(coord_merged,sizeof(float4)*(n+1));
  iris::DMem dev_veloc(veloc,sizeof(float4)*(n+1));
  iris::DMem dev_nbond(nbond,sizeof(int)*(n+1));
  iris::DMem dev_bonds(bonds,sizeof(int2)*(n+n+1));
  iris::DMem dev_bond_r0(bond_r0,sizeof(r64)*(n+n+1));
  iris::DMem dev_force_x(force_x,sizeof(r64)*(n+1));
  iris::DMem dev_force_y(force_y,sizeof(r64)*(n+1));
  iris::DMem dev_force_z(force_z,sizeof(r64)*(n+1));
  iris::DMem dev_bond_l0(bond_l0,sizeof(r32)*(n+1));
  iris::DMem dev_temp(temp,sizeof(r32)*(n+1));
  iris::DMem dev_mu_targ(mu_targ,sizeof(r32)*(n+1));
  iris::DMem dev_qp(qp,sizeof(r32)*(n+1));
  iris::DMem dev_gamc(gamc,sizeof(r32)*(n+1));
  iris::DMem dev_gamt(gamt,sizeof(r32)*(n+1));
  iris::DMem dev_sigc(sigc,sizeof(r32)*(n+1));
  iris::DMem dev_sigt(sigt,sizeof(r32)*(n+1));
  size_t lws=128, gws=((n+127)/128)*lws;
  auto start = std::chrono::steady_clock::now();

  // note the outputs are not reset for each run
  for (i = 0; i < repeat; i++) {
    iris::Task task;
    void* args[]={&dev_force_x,&dev_force_y,&dev_force_z,&dev_coord_merged,&dev_veloc,&dev_nbond,&dev_bonds,&dev_bond_r0,&dev_temp,&dev_bond_l0,&dev_mu_targ,&dev_qp,&dev_gamc,&dev_gamt,&dev_sigc,&dev_sigt,&period,&padding,&n_type,&n};
    int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,sizeof(period),sizeof(padding),sizeof(n_type),sizeof(n)};
    iris_check(task.kernel("bond_wlcpowallvisc",1,nullptr,&gws,&lws,20,args,info),"kernel");
    iris_check(task.submit(device,nullptr,true),"HIP task");
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", time * 1e-3f / repeat);

  iris::Task flush;
  iris_check(flush.flush_out(dev_force_x),"flush x");
  iris_check(flush.flush_out(dev_force_y),"flush y");
  iris_check(flush.flush_out(dev_force_z),"flush z");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");
  bool finite=true;

  // no NaN values in the outputs
  for (i = 0; i < n+1; i++) {
    bool r = (std::isnan(force_x[i]) || std::isnan(force_y[i]) || std::isnan(force_z[i]));
    finite &= std::isfinite(force_x[i]) && std::isfinite(force_y[i]) && std::isfinite(force_z[i]);
    if (r) printf("There are NaN numbers at index %d\n", i);
  }

  double force_x_sum = 0, force_y_sum = 0, force_z_sum = 0;
  for (i = 0; i < n+1; i++) {
    force_x_sum += force_x[i];
    force_y_sum += force_y[i];
    force_z_sum += force_z[i];
  }
  // values are meaningless, but they should be consistent across devices
  printf("checksum: forceX=%lf forceY=%lf forceZ=%lf\n",
         force_x_sum/(n+1), force_y_sum/(n+1), force_z_sum/(n+1));
  
#ifdef DEBUG
  for (i = 0; i < 16; i++) {
    printf("%d %lf %lf %lf\n", i, force_x[i], force_y[i], force_z[i]);
  }
#endif

  free(coord_merged);
  free(veloc);
  free(force_x);
  free(force_y);
  free(force_z);
  free(nbond);
  free(bonds);
  free(bond_r0);
  free(bond_l0);
  free(temp);
  free(mu_targ);
  free(qp);
  free(gamc);
  free(gamt);
  free(sigc);
  free(sigt);

  iris_check(platform.finalize(),"finalize");
  return finite ? 0 : 1;
}
