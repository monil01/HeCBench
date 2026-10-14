#include <iris/iris.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
extern "C" {
#include "fem.h"
}
#undef min
#undef max

static iris::DMem *c_LIFT, *c_DrDsDt, *c_surfinfo, *c_vgeo, *c_Q, *c_partQ, *c_rhsQ, *c_resQ, *c_tmp;
static std::vector<iris::DMem*> buffers;
static void require(int e) { if(e!=IRIS_SUCCESS) throw std::runtime_error("IRIS failure"); }
static iris::DMem *make_mem(size_t bytes) {
  bytes=std::max(bytes,sizeof(float));
  void *host=std::calloc(1,bytes);
  if(!host) throw std::bad_alloc();
  auto *mem=new iris::DMem(host,bytes);
  buffers.push_back(mem);
  return mem;
}
static void upload(iris::DMem *mem,const void *data,size_t bytes) {
  std::memcpy(mem->host(),data,bytes);
  require(mem->update(mem->host()));
}
static void download(iris::DMem *mem,void *data,size_t bytes) {
  iris::Task task;
  require(task.flush_out(*mem)); require(task.submit(iris_gpu,nullptr,1));
  std::memcpy(data,mem->host(),bytes);
}
static void release_buffers() {
  for(auto *mem:buffers) { void *host=mem->host(); delete mem; std::free(host); }
  buffers.clear();
}
static void launch(const char *kernel,size_t blocks,size_t threads,int count,void **params,int *info,iris::DMem *out=nullptr) {
  if(!blocks) return;
  iris::Task task;
  size_t global[]={blocks*threads},local[]={threads};
  require(task.kernel(kernel,1,nullptr,global,local,count,params,info));
  if(out) require(task.flush_out(*out));
  require(task.submit(iris_gpu,nullptr,1));
}
extern "C" {
double InitGPU3d(Mesh *mesh, int Nfields){

  /* Q  */
  int sz = mesh->K*(BSIZE)*p_Nfields*sizeof(float);

  float *f_Q = (float*) calloc(mesh->K*BSIZE*p_Nfields, sizeof(float));
  c_Q = make_mem(sz);
  c_rhsQ = make_mem(sz);
  c_resQ = make_mem(sz);
  c_tmp = make_mem(sz);
  upload(c_Q, f_Q, sz);
  upload(c_rhsQ, f_Q, sz);
  upload(c_resQ, f_Q, sz);
  upload(c_tmp, f_Q, sz);

  sz = mesh->parNtotalout*sizeof(float);
  c_partQ = make_mem(sz);

  /*  LIFT  */
   sz = p_Np*(p_Nfp)*p_Nfaces*sizeof(float);
#if 0
   float *f_LIFT = (float*) malloc(sz);
   int skL = 0;
   for(int m=0;m<p_Nfp*p_Nfaces;++m){
     for(int n=0;n<p_Np;++n){
       f_LIFT[skL++] = d_LIFT[n+p_Np*m];
     }
   }
#else
   float *f_LIFT = (float*) malloc(sz);
   int skL = 0;
   for(int m=0;m<p_Nfp;++m){
     for(int n=0;n<p_Np;++n){
       for(int f=0;f<p_Nfaces;++f){
	 f_LIFT[skL++] = mesh->LIFT[0][p_Nfp*p_Nfaces*n+(f+p_Nfaces*m)];
       }
     }
   }
#endif
   c_LIFT = make_mem(sz);
   upload(c_LIFT, f_LIFT, sz);

   /* Bind the array to the texture */

   /* DrDsDt */
   sz = BSIZE*BSIZE*4*sizeof(float);

   float* h_DrDsDt = (float*) calloc(BSIZE*BSIZE, (4*sizeof(float)));
   int sk = 0;
   /* note transposed arrays to avoid "bank conflicts" */
   for(int n=0;n<p_Np;++n){
     for(int m=0;m<p_Np;++m){
       h_DrDsDt[4*(m+n*BSIZE)+0] = mesh->Dr[0][n+m*p_Np];
       h_DrDsDt[4*(m+n*BSIZE)+1] = mesh->Ds[0][n+m*p_Np];
       h_DrDsDt[4*(m+n*BSIZE)+2] = mesh->Dt[0][n+m*p_Np];
     }
   }

   c_DrDsDt = make_mem(sz);
   upload(c_DrDsDt, h_DrDsDt, sz);

   /* Bind the array to the texture */

   free(h_DrDsDt);

   /* vgeo */
   double drdx, dsdx, dtdx;
   double drdy, dsdy, dtdy;
   double drdz, dsdz, dtdz, J;
   float *vgeo = (float*) calloc(12*mesh->K, sizeof(float));

   for(int k=0;k<mesh->K;++k){
     GeometricFactors3d(mesh, k,
			&drdx, &dsdx, &dtdx,
			&drdy, &dsdy, &dtdy,
			&drdz, &dsdz, &dtdz, &J);

     vgeo[k*12+0] = drdx; vgeo[k*12+1] = drdy; vgeo[k*12+2] = drdz;
     vgeo[k*12+4] = dsdx; vgeo[k*12+5] = dsdy; vgeo[k*12+6] = dsdz;
     vgeo[k*12+8] = dtdx; vgeo[k*12+9] = dtdy; vgeo[k*12+10] = dtdz;

   }

   sz = mesh->K*12*sizeof(float);
   c_vgeo = make_mem(sz);
   upload(c_vgeo, vgeo, sz);

   /* surfinfo (vmapM, vmapP, Fscale, Bscale, nx, ny, nz, 0) */
   sz = mesh->K*p_Nfp*p_Nfaces*7*sizeof(float);
   float* h_surfinfo = (float*) malloc(sz);

   /* local-local info */
   sk = 0;
   int skP = -1;
   double *nxk = BuildVector(mesh->Nfaces);
   double *nyk = BuildVector(mesh->Nfaces);
   double *nzk = BuildVector(mesh->Nfaces);
   double *sJk = BuildVector(mesh->Nfaces);

   double dt = 1e6;

   for(int k=0;k<mesh->K;++k){

     GeometricFactors3d(mesh, k,
			&drdx, &dsdx, &dtdx,
			&drdy, &dsdy, &dtdy,
			&drdz, &dsdz, &dtdz, &J);

     Normals3d(mesh, k, nxk, nyk, nzk, sJk);

     for(int f=0;f<mesh->Nfaces;++f){

	     dt = std::min(dt, J/sJk[f]);

       for(int m=0;m<p_Nfp;++m){
	 int n = m + f*p_Nfp + p_Nfp*p_Nfaces*k;
	 int idM = mesh->vmapM[n];
	 int idP = mesh->vmapP[n];
	 int  nM = idM%p_Np;
	 int  nP = idP%p_Np;
	 int  kM = (idM-nM)/p_Np;
	 int  kP = (idP-nP)/p_Np;
	 idM = nM + Nfields*BSIZE*kM;
	 idP = nP + Nfields*BSIZE*kP;

	 /* stub resolve some other way */
	 if(mesh->vmapP[n]<0){
	   idP = mesh->vmapP[n]; /* -ve numbers */
	 }

	 sk = 7*p_Nfp*p_Nfaces*k+m+f*p_Nfp;
	 h_surfinfo[sk + 0*p_Nfp*p_Nfaces] = idM;
	 h_surfinfo[sk + 1*p_Nfp*p_Nfaces] = idP;
	 h_surfinfo[sk + 2*p_Nfp*p_Nfaces] = sJk[f]/(2.*J);
	 h_surfinfo[sk + 3*p_Nfp*p_Nfaces] = (idM==idP)?-1.:1.;
	 h_surfinfo[sk + 4*p_Nfp*p_Nfaces] = nxk[f];
	 h_surfinfo[sk + 5*p_Nfp*p_Nfaces] = nyk[f];
	 h_surfinfo[sk + 6*p_Nfp*p_Nfaces] = nzk[f];
       }
     }
   }

   c_surfinfo = make_mem(sz);
   upload(c_surfinfo, h_surfinfo, sz);

   free(h_surfinfo);

   /* Host index list becomes a dmem argument in the halo extraction task. */
   mesh->c_parmapOUT = mesh->parmapOUT;

   return dt;
}




void iris_copy_part(Mesh *mesh) { upload(c_partQ,mesh->f_inQ,mesh->parNtotalout*sizeof(float)); }
void iris_sync(void) {}
void MaxwellsKernel3d(Mesh *mesh,float fa,float fb,float fdt) {
  MaxwellsMPISend3d(mesh);
  void *vol[]={c_rhsQ,c_Q,c_vgeo,c_DrDsDt};
  /* Volume writes active nodes; preserve the initialized padded lanes. */
  int vi[]={iris_rw,iris_r,iris_r,iris_r};
  launch("MaxwellsGPU_VOL_Kernel3D",mesh->K,p_Np,4,vol,vi);
  MaxwellsMPIRecv3d(mesh,nullptr);
  void *surf[]={c_rhsQ,c_Q,c_partQ,c_surfinfo,c_LIFT};
  int si[]={iris_rw,iris_r,iris_r,iris_r,iris_r};
  launch("MaxwellsGPU_SURF_Kernel3D",mesh->K,std::max(p_Nfp*p_Nfaces,p_Np),5,surf,si);
  int total=mesh->K*BSIZE*p_Nfields;
  void *rk[]={&total,c_resQ,c_rhsQ,c_Q,&fa,&fb,&fdt};
  int ri[]={sizeof(total),iris_rw,iris_r,iris_rw,sizeof(fa),sizeof(fb),sizeof(fdt)};
  launch("MaxwellsGPU_RK_Kernel3D",(total+255)/256,256,7,rk,ri);
}
void gpu_set_data3d(int K,
		  double *d_Hx, double *d_Hy, double *d_Hz,
		  double *d_Ex, double *d_Ey, double *d_Ez)
{
  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));

  /* also load into usual data matrices */

  for(int k=0;k<K;++k){
    int gk = k;
    for(int n=0;n<p_Np;++n)
      f_Q[n        +k*BSIZE*p_Nfields] = d_Hx[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n  +BSIZE+k*BSIZE*p_Nfields] = d_Hy[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+2*BSIZE+k*BSIZE*p_Nfields] = d_Hz[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+3*BSIZE+k*BSIZE*p_Nfields] = d_Ex[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+4*BSIZE+k*BSIZE*p_Nfields] = d_Ey[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+5*BSIZE+k*BSIZE*p_Nfields] = d_Ez[n+gk*p_Np];
  }

  upload(c_Q, f_Q, BSIZE*K*p_Nfields*sizeof(float));

  free(f_Q);
}

void gpu_get_data3d(int K,
		  double *d_Hx, double *d_Hy, double *d_Hz,
		  double *d_Ex, double *d_Ey, double *d_Ez){

  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));

  download(c_Q, f_Q, K*BSIZE*p_Nfields*sizeof(float));

  /* also load into usual data matrices */

  for(int k=0;k<K;++k) {
    int gk = k;
    for(int n=0;n<p_Np;++n)
      d_Hx[n+gk*p_Np] = f_Q[n        +k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Hy[n+gk*p_Np] = f_Q[n  +BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Hz[n+gk*p_Np] = f_Q[n+2*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ex[n+gk*p_Np] = f_Q[n+3*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ey[n+gk*p_Np] = f_Q[n+4*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ez[n+gk*p_Np] = f_Q[n+5*BSIZE+k*BSIZE*p_Nfields];
  }

  free(f_Q);
  release_buffers();
}


void get_partial_gpu_data3d(int total,int *index,float *output) {
  if(!total) return;
  iris::DMem indices(index,total*sizeof(int));
  void *params[]={&total,&indices,c_tmp,c_Q};
  int info[]={sizeof(total),iris_r,iris_rw,iris_r};
  launch("partial_get_kernel3d",(total+255)/256,256,4,params,info,c_tmp);
  std::memcpy(output,c_tmp->host(),total*sizeof(float));
}
}
