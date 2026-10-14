#include "mpi.h"
#include <iris/iris.hpp>
extern "C" {
#include "fem.h"
}
#undef min
#undef max

int main(int argc, char **argv){

  if(argc!=2) { fprintf(stderr,"Usage: %s mesh.neu\n",argv[0]); return 1; }
  Mesh *mesh;
  int procid, nprocs, maxNv;
  int k,n, sk=0;
  double minEz, maxEz, gminEz, gmaxEz;

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* assign gpu */
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#ifdef CUDA
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) MPI_Abort(MPI_COMM_WORLD,2);
#endif

  /* (parallel) read part of fem mesh from file */
  mesh = ReadMesh3d(argv[1]);

  /* perform load balancing */
  LoadBalance3d(mesh);

  /* find element-element connectivity */
  FacePair3d(mesh, &maxNv);

  /* perform start up */
  StartUp3d(mesh);

  /* field storage (double) */
  double *Hx = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hy = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hz = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ex = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ey = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ez = (double*) calloc(mesh->K*p_Np, sizeof(double));

  /* initial conditions */
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n) {
      Hx[sk] = 0;
      Hy[sk] = 0;
      Hz[sk] = 0;
      Ex[sk] = 0;
      Ey[sk] = 0;
      Ez[sk] = cos(M_PI*mesh->x[k][n])*cos(M_PI*mesh->y[k][n])*cos(M_PI*mesh->z[k][n]);
      ++sk;
    }
  }

  double dt, gdt;

#ifdef CUDA
  /* initialize GPU info */
  dt = InitGPU3d(mesh, p_Nfields);

  /* load data onto GPU */
  gpu_set_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);
#else
  /* initialize GPU info */
  dt = InitCPU3d(mesh, p_Nfields);

  /* load data onto CPU float storage */
  cpu_set_data3d(mesh, Hx, Hy, Hz, Ex, Ey, Ez);
#endif

  MPI_Allreduce(&dt, &gdt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  dt = .5*gdt/((p_N+1)*(p_N+1));

  if(mesh->procid==0)
    printf("dt = %f\n", dt);

  double FinalTime = .00050;

  /* solve */
  MaxwellsRun3d(mesh, FinalTime, dt);

#ifdef CUDA
  /* unload data from GPU */
  gpu_get_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);
#else
  cpu_get_data3d(mesh, Hx, Hy, Hz, Ex, Ey, Ez);
#endif

  const char *dir=getenv("MINIDGS_VERIFY_DIR");
  if(dir) {
    char path[4096]; snprintf(path,sizeof(path),"%s/rank_%d.bin",dir,procid);
    FILE *out=fopen(path,"wb");
    if(!out) MPI_Abort(MPI_COMM_WORLD,3);
    double *fields[]={Hx,Hy,Hz,Ex,Ey,Ez};
    for(int field=0;field<6;++field)
      if(fwrite(fields[field],sizeof(double),mesh->K*p_Np,out)!=(size_t)(mesh->K*p_Np))
        MPI_Abort(MPI_COMM_WORLD,3);
    fclose(out);
  }

  /* find maximum & minimum values for Ez */
  minEz=Ez[0], maxEz=Ez[0];

  for(n=0;n<mesh->K*p_Np;++n) {
    minEz = (minEz>Ez[n])?Ez[n]:minEz;
    maxEz = (maxEz<Ez[n])?Ez[n]:maxEz;
    printf("%e\n", Ez[n]);
  }

  MPI_Reduce(&minEz, &gminEz, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxEz, &gmaxEz, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(procid==0)
    printf("t=%f Ez in [ %f, %f ] \n", FinalTime, gminEz, gmaxEz );

  free(Hx);
  free(Hy);
  free(Hz);
  free(Ex);
  free(Ey);
  free(Ez);

  /* stop MPI */
  if(platform.finalize()!=IRIS_SUCCESS) MPI_Abort(MPI_COMM_WORLD,2);
  MPI_Finalize();

  return 0;
}
