#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "study_setup.h"
#ifdef CPU_REFERENCE
#include "source_cpu_math.h"
static inline
void TimeIncrement(Domain& domain)
{
  Real_t targetdt = domain.stoptime() - domain.time() ;

  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
    Real_t ratio ;
    Real_t olddt = domain.deltatime() ;

    /* This will require a reduction in parallel */
    Real_t gnewdt = Real_t(1.0e+20) ;
    Real_t newdt ;
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0) ;
    }
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
    }

    newdt = gnewdt;

    ratio = newdt / olddt ;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt ;
      }
      else if (ratio > domain.deltatimemultub()) {
        newdt = olddt*domain.deltatimemultub() ;
      }
    }

    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax() ;
    }
    domain.deltatime() = newdt ;
  }

  /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
  if ((targetdt > domain.deltatime()) &&
      (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt ;
  }

  domain.time() += domain.deltatime() ;

  ++domain.cycle() ;
}


/******************************************/


/******************************************/

static inline
void CalcCourantConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist,
    Real_t qqc, Real_t& dtcourant)
{
  Index_t threads = NT;
  static Index_t *courant_elem_per_thread;
  static Real_t *dtcourant_per_thread;
  static bool first = true;
  if (first) {
    courant_elem_per_thread = new Index_t[threads];
    dtcourant_per_thread = new Real_t[threads];
    first = false;
  }

  //#pragma omp parallel firstprivate(length, qqc)
  {
    Real_t   qqc2 = Real_t(64.0) * qqc * qqc ;
    Real_t   dtcourant_tmp = dtcourant;
    Index_t  courant_elem  = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;
        Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

        if ( domain.vdov(indx) < Real_t(0.) ) {
          dtf = dtf
            + qqc2 * domain.arealg(indx) * domain.arealg(indx)
            * domain.vdov(indx) * domain.vdov(indx) ;
        }

        dtf = SQRT(dtf) ;
        dtf = domain.arealg(indx) / dtf ;

        if (domain.vdov(indx) != Real_t(0.)) {
          if ( dtf < dtcourant_tmp ) {
            dtcourant_tmp = dtf ;
            courant_elem  = indx ;
          }
        }
      }

      dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
      courant_elem_per_thread[thread_num] = courant_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
      dtcourant_per_thread[0]    = dtcourant_per_thread[i];
      courant_elem_per_thread[0] = courant_elem_per_thread[i];
    }
  }

  if (courant_elem_per_thread[0] != -1) {
    dtcourant = dtcourant_per_thread[0] ;
  }

  return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
  Index_t threads = NT;
  static Index_t *hydro_elem_per_thread;
  static Real_t *dthydro_per_thread;
  static bool first = true;
  if (first) {
    hydro_elem_per_thread = new Index_t[threads];
    dthydro_per_thread = new Real_t[threads];
    first = false;
  }

  //#pragma omp parallel firstprivate(length, dvovmax)
  {
    Real_t dthydro_tmp = dthydro ;
    Index_t hydro_elem = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;

        if (domain.vdov(indx) != Real_t(0.)) {
          Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx))+Real_t(1.e-20)) ;

          if ( dthydro_tmp > dtdvov ) {
            dthydro_tmp = dtdvov ;
            hydro_elem = indx ;
          }
        }
      }

      dthydro_per_thread[thread_num]    = dthydro_tmp ;
      hydro_elem_per_thread[thread_num] = hydro_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
      dthydro_per_thread[0]    = dthydro_per_thread[i];
      hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
    }
  }

  if (hydro_elem_per_thread[0] != -1) {
    dthydro =  dthydro_per_thread[0] ;
  }

  return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain) {

  // Initialize conditions to a very large value
  domain.dtcourant() = 1.0e+20;
  domain.dthydro() = 1.0e+20;

  for (Index_t r=0 ; r < domain.numReg() ; ++r) {
    /* evaluate time constraint */
    CalcCourantConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.qqc(),
        domain.dtcourant()) ;

    /* check hydro constraint */
    CalcHydroConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.dvovmax(),
        domain.dthydro()) ;
  }
}
/******************************************/


/******************************************/


#else
#define main original_lulesh_main
#include "../lulesh-cuda/lulesh.cu"
#undef main
#endif
static void ck(int e){if(e){fprintf(stderr,"CUDA error %d\n",e);exit(2);}}
static std::vector<double> result;
static constexpr int G=64;
template<class T>struct Buf{
 std::vector<T> initial,host;T*ptr=nullptr;
 Buf(){}
 void init(FILE*f,size_t n){initial.assign(n+2*G,T(-987654321.125));fread(initial.data()+G,sizeof(T),n,f);host=initial;
#ifdef CPU_REFERENCE
 ptr=host.data()+G;
#else
 T*raw;ck(cudaMalloc((void**)&raw,sizeof(T)*host.size()));ptr=raw+G;ck(cudaMemcpy(raw,host.data(),sizeof(T)*host.size(),cudaMemcpyHostToDevice));
#endif
 }
 void reset(){host=initial;
#ifdef CPU_REFERENCE
 ptr=host.data()+G;
#else
 ck(cudaMemcpy(ptr-G,host.data(),sizeof(T)*host.size(),cudaMemcpyHostToDevice));
#endif
 }
 void capture(){
#ifndef CPU_REFERENCE
 ck(cudaMemcpy(host.data(),ptr-G,sizeof(T)*host.size(),cudaMemcpyDeviceToHost));
#endif
 for(T x:host)result.push_back(double(x));}
 ~Buf(){
#ifndef CPU_REFERENCE
 if(ptr)cudaFree(ptr-G);
#endif
 }
};
struct State{Domain d;int nx,steps,mode;double seconds=0;std::vector<double>params;std::vector<int>regions;
Buf<Real_t> b_x;Real_t*d_x;
Buf<Real_t> b_y;Real_t*d_y;
Buf<Real_t> b_z;Real_t*d_z;
Buf<Real_t> b_fx;Real_t*d_fx;
Buf<Real_t> b_fy;Real_t*d_fy;
Buf<Real_t> b_fz;Real_t*d_fz;
Buf<Real_t> b_xd;Real_t*d_xd;
Buf<Real_t> b_yd;Real_t*d_yd;
Buf<Real_t> b_zd;Real_t*d_zd;
Buf<Real_t> b_xdd;Real_t*d_xdd;
Buf<Real_t> b_ydd;Real_t*d_ydd;
Buf<Real_t> b_zdd;Real_t*d_zdd;
Buf<Index_t> b_nodelist;Index_t*d_nodelist;
Buf<Real_t> b_determ;Real_t*d_determ;
Buf<Real_t> b_fx_elem;Real_t*d_fx_elem;
Buf<Real_t> b_fy_elem;Real_t*d_fy_elem;
Buf<Real_t> b_fz_elem;Real_t*d_fz_elem;
Buf<Real_t> b_dvdx;Real_t*d_dvdx;
Buf<Real_t> b_dvdy;Real_t*d_dvdy;
Buf<Real_t> b_dvdz;Real_t*d_dvdz;
Buf<Real_t> b_x8n;Real_t*d_x8n;
Buf<Real_t> b_y8n;Real_t*d_y8n;
Buf<Real_t> b_z8n;Real_t*d_z8n;
Buf<Real_t> b_sigxx;Real_t*d_sigxx;
Buf<Real_t> b_sigyy;Real_t*d_sigyy;
Buf<Real_t> b_sigzz;Real_t*d_sigzz;
Buf<Real_t> b_delv_xi;Real_t*d_delv_xi;
Buf<Real_t> b_delx_xi;Real_t*d_delx_xi;
Buf<Real_t> b_delv_eta;Real_t*d_delv_eta;
Buf<Real_t> b_delx_eta;Real_t*d_delx_eta;
Buf<Real_t> b_delv_zeta;Real_t*d_delv_zeta;
Buf<Real_t> b_delx_zeta;Real_t*d_delx_zeta;
Buf<Real_t> b_p;Real_t*d_p;
Buf<Real_t> b_q;Real_t*d_q;
Buf<Real_t> b_volo;Real_t*d_volo;
Buf<Real_t> b_v;Real_t*d_v;
Buf<int> b_vol_error;int*d_vol_error;
Buf<Index_t> b_nodeElemStart;Index_t*d_nodeElemStart;
Buf<Index_t> b_nodeElemCornerList;Index_t*d_nodeElemCornerList;
Buf<Real_t> b_gamma;Real_t*d_gamma;
Buf<Real_t> b_ss;Real_t*d_ss;
Buf<Real_t> b_elemMass;Real_t*d_elemMass;
Buf<Real_t> b_nodalMass;Real_t*d_nodalMass;
Buf<Index_t> b_symmX;Index_t*d_symmX;
Buf<Index_t> b_symmY;Index_t*d_symmY;
Buf<Index_t> b_symmZ;Index_t*d_symmZ;
Buf<Real_t> b_vdov;Real_t*d_vdov;
Buf<Real_t> b_delv;Real_t*d_delv;
Buf<Real_t> b_arealg;Real_t*d_arealg;
Buf<Real_t> b_dxx;Real_t*d_dxx;
Buf<Real_t> b_dyy;Real_t*d_dyy;
Buf<Real_t> b_dzz;Real_t*d_dzz;
Buf<Real_t> b_vnew;Real_t*d_vnew;
Buf<Index_t> b_lzetam;Index_t*d_lzetam;
Buf<Index_t> b_lzetap;Index_t*d_lzetap;
Buf<Index_t> b_letap;Index_t*d_letap;
Buf<Index_t> b_letam;Index_t*d_letam;
Buf<Index_t> b_lxip;Index_t*d_lxip;
Buf<Index_t> b_lxim;Index_t*d_lxim;
Buf<Index_t> b_elemBC;Index_t*d_elemBC;
Buf<Real_t> b_ql;Real_t*d_ql;
Buf<Real_t> b_qq;Real_t*d_qq;
Buf<Real_t> b_e;Real_t*d_e;
Buf<Index_t> b_elemRep;Index_t*d_elemRep;
Buf<Index_t> b_elemElem;Index_t*d_elemElem;
State(int index,int n,int ns,int md):d(1,0,0,0,n,1,11,1,1),nx(n),steps(ns),mode(md){prepare_domain(d,md);int numNode=d.numNode(),numElem=d.numElem(),numElem8=8*numElem,numNodeBC=(nx+1)*(nx+1),len1=numNode+1,len2=numElem8;FILE*f=fopen(("case"+std::to_string(index)+".bin").c_str(),"rb");if(!f)exit(3);
b_x.init(f,numNode);d_x=b_x.ptr;
b_y.init(f,numNode);d_y=b_y.ptr;
b_z.init(f,numNode);d_z=b_z.ptr;
b_fx.init(f,numNode);d_fx=b_fx.ptr;
b_fy.init(f,numNode);d_fy=b_fy.ptr;
b_fz.init(f,numNode);d_fz=b_fz.ptr;
b_xd.init(f,numNode);d_xd=b_xd.ptr;
b_yd.init(f,numNode);d_yd=b_yd.ptr;
b_zd.init(f,numNode);d_zd=b_zd.ptr;
b_xdd.init(f,numNode);d_xdd=b_xdd.ptr;
b_ydd.init(f,numNode);d_ydd=b_ydd.ptr;
b_zdd.init(f,numNode);d_zdd=b_zdd.ptr;
b_nodelist.init(f,numElem8);d_nodelist=b_nodelist.ptr;
b_determ.init(f,numElem);d_determ=b_determ.ptr;
b_fx_elem.init(f,numElem8);d_fx_elem=b_fx_elem.ptr;
b_fy_elem.init(f,numElem8);d_fy_elem=b_fy_elem.ptr;
b_fz_elem.init(f,numElem8);d_fz_elem=b_fz_elem.ptr;
b_dvdx.init(f,numElem8);d_dvdx=b_dvdx.ptr;
b_dvdy.init(f,numElem8);d_dvdy=b_dvdy.ptr;
b_dvdz.init(f,numElem8);d_dvdz=b_dvdz.ptr;
b_x8n.init(f,numElem8);d_x8n=b_x8n.ptr;
b_y8n.init(f,numElem8);d_y8n=b_y8n.ptr;
b_z8n.init(f,numElem8);d_z8n=b_z8n.ptr;
b_sigxx.init(f,numElem);d_sigxx=b_sigxx.ptr;
b_sigyy.init(f,numElem);d_sigyy=b_sigyy.ptr;
b_sigzz.init(f,numElem);d_sigzz=b_sigzz.ptr;
b_delv_xi.init(f,numElem);d_delv_xi=b_delv_xi.ptr;
b_delx_xi.init(f,numElem);d_delx_xi=b_delx_xi.ptr;
b_delv_eta.init(f,numElem);d_delv_eta=b_delv_eta.ptr;
b_delx_eta.init(f,numElem);d_delx_eta=b_delx_eta.ptr;
b_delv_zeta.init(f,numElem);d_delv_zeta=b_delv_zeta.ptr;
b_delx_zeta.init(f,numElem);d_delx_zeta=b_delx_zeta.ptr;
b_p.init(f,numElem);d_p=b_p.ptr;
b_q.init(f,numElem);d_q=b_q.ptr;
b_volo.init(f,numElem);d_volo=b_volo.ptr;
b_v.init(f,numElem);d_v=b_v.ptr;
b_vol_error.init(f,1);d_vol_error=b_vol_error.ptr;
b_nodeElemStart.init(f,len1);d_nodeElemStart=b_nodeElemStart.ptr;
b_nodeElemCornerList.init(f,numElem8);d_nodeElemCornerList=b_nodeElemCornerList.ptr;
b_gamma.init(f,32);d_gamma=b_gamma.ptr;
b_ss.init(f,numElem);d_ss=b_ss.ptr;
b_elemMass.init(f,numElem);d_elemMass=b_elemMass.ptr;
b_nodalMass.init(f,numNode);d_nodalMass=b_nodalMass.ptr;
b_symmX.init(f,numNodeBC);d_symmX=b_symmX.ptr;
b_symmY.init(f,numNodeBC);d_symmY=b_symmY.ptr;
b_symmZ.init(f,numNodeBC);d_symmZ=b_symmZ.ptr;
b_vdov.init(f,numElem);d_vdov=b_vdov.ptr;
b_delv.init(f,numElem);d_delv=b_delv.ptr;
b_arealg.init(f,numElem);d_arealg=b_arealg.ptr;
b_dxx.init(f,numElem);d_dxx=b_dxx.ptr;
b_dyy.init(f,numElem);d_dyy=b_dyy.ptr;
b_dzz.init(f,numElem);d_dzz=b_dzz.ptr;
b_vnew.init(f,numElem);d_vnew=b_vnew.ptr;
b_lzetam.init(f,numElem);d_lzetam=b_lzetam.ptr;
b_lzetap.init(f,numElem);d_lzetap=b_lzetap.ptr;
b_letap.init(f,numElem);d_letap=b_letap.ptr;
b_letam.init(f,numElem);d_letam=b_letam.ptr;
b_lxip.init(f,numElem);d_lxip=b_lxip.ptr;
b_lxim.init(f,numElem);d_lxim=b_lxim.ptr;
b_elemBC.init(f,numElem);d_elemBC=b_elemBC.ptr;
b_ql.init(f,numElem);d_ql=b_ql.ptr;
b_qq.init(f,numElem);d_qq=b_qq.ptr;
b_e.init(f,numElem);d_e=b_e.ptr;
b_elemRep.init(f,numElem);d_elemRep=b_elemRep.ptr;
b_elemElem.init(f,numElem);d_elemElem=b_elemElem.ptr;
fclose(f);f=fopen(("params"+std::to_string(index)+".bin").c_str(),"rb");params.resize(26);fread(params.data(),8,params.size(),f);fclose(f);f=fopen(("regions"+std::to_string(index)+".bin").c_str(),"rb");for(int r=0;r<11;r++){int n;fread(&n,4,1,f);regions.push_back(n);for(int i=0;i<n;i++){int v;fread(&v,4,1,f);regions.push_back(v);}}fclose(f);}
void reset(){seconds=0;d.time()=params[24];d.cycle()=0;d.deltatime()=params[25];d.dtcourant()=params[21];d.dthydro()=params[22];
b_x.reset();d_x=b_x.ptr;
b_y.reset();d_y=b_y.ptr;
b_z.reset();d_z=b_z.ptr;
b_fx.reset();d_fx=b_fx.ptr;
b_fy.reset();d_fy=b_fy.ptr;
b_fz.reset();d_fz=b_fz.ptr;
b_xd.reset();d_xd=b_xd.ptr;
b_yd.reset();d_yd=b_yd.ptr;
b_zd.reset();d_zd=b_zd.ptr;
b_xdd.reset();d_xdd=b_xdd.ptr;
b_ydd.reset();d_ydd=b_ydd.ptr;
b_zdd.reset();d_zdd=b_zdd.ptr;
b_nodelist.reset();d_nodelist=b_nodelist.ptr;
b_determ.reset();d_determ=b_determ.ptr;
b_fx_elem.reset();d_fx_elem=b_fx_elem.ptr;
b_fy_elem.reset();d_fy_elem=b_fy_elem.ptr;
b_fz_elem.reset();d_fz_elem=b_fz_elem.ptr;
b_dvdx.reset();d_dvdx=b_dvdx.ptr;
b_dvdy.reset();d_dvdy=b_dvdy.ptr;
b_dvdz.reset();d_dvdz=b_dvdz.ptr;
b_x8n.reset();d_x8n=b_x8n.ptr;
b_y8n.reset();d_y8n=b_y8n.ptr;
b_z8n.reset();d_z8n=b_z8n.ptr;
b_sigxx.reset();d_sigxx=b_sigxx.ptr;
b_sigyy.reset();d_sigyy=b_sigyy.ptr;
b_sigzz.reset();d_sigzz=b_sigzz.ptr;
b_delv_xi.reset();d_delv_xi=b_delv_xi.ptr;
b_delx_xi.reset();d_delx_xi=b_delx_xi.ptr;
b_delv_eta.reset();d_delv_eta=b_delv_eta.ptr;
b_delx_eta.reset();d_delx_eta=b_delx_eta.ptr;
b_delv_zeta.reset();d_delv_zeta=b_delv_zeta.ptr;
b_delx_zeta.reset();d_delx_zeta=b_delx_zeta.ptr;
b_p.reset();d_p=b_p.ptr;
b_q.reset();d_q=b_q.ptr;
b_volo.reset();d_volo=b_volo.ptr;
b_v.reset();d_v=b_v.ptr;
b_vol_error.reset();d_vol_error=b_vol_error.ptr;
b_nodeElemStart.reset();d_nodeElemStart=b_nodeElemStart.ptr;
b_nodeElemCornerList.reset();d_nodeElemCornerList=b_nodeElemCornerList.ptr;
b_gamma.reset();d_gamma=b_gamma.ptr;
b_ss.reset();d_ss=b_ss.ptr;
b_elemMass.reset();d_elemMass=b_elemMass.ptr;
b_nodalMass.reset();d_nodalMass=b_nodalMass.ptr;
b_symmX.reset();d_symmX=b_symmX.ptr;
b_symmY.reset();d_symmY=b_symmY.ptr;
b_symmZ.reset();d_symmZ=b_symmZ.ptr;
b_vdov.reset();d_vdov=b_vdov.ptr;
b_delv.reset();d_delv=b_delv.ptr;
b_arealg.reset();d_arealg=b_arealg.ptr;
b_dxx.reset();d_dxx=b_dxx.ptr;
b_dyy.reset();d_dyy=b_dyy.ptr;
b_dzz.reset();d_dzz=b_dzz.ptr;
b_vnew.reset();d_vnew=b_vnew.ptr;
b_lzetam.reset();d_lzetam=b_lzetam.ptr;
b_lzetap.reset();d_lzetap=b_lzetap.ptr;
b_letap.reset();d_letap=b_letap.ptr;
b_letam.reset();d_letam=b_letam.ptr;
b_lxip.reset();d_lxip=b_lxip.ptr;
b_lxim.reset();d_lxim=b_lxim.ptr;
b_elemBC.reset();d_elemBC=b_elemBC.ptr;
b_ql.reset();d_ql=b_ql.ptr;
b_qq.reset();d_qq=b_qq.ptr;
b_e.reset();d_e=b_e.ptr;
b_elemRep.reset();d_elemRep=b_elemRep.ptr;
b_elemElem.reset();d_elemElem=b_elemElem.ptr;
}
void capture(bool initial){
b_x.capture();
b_y.capture();
b_z.capture();
b_fx.capture();
b_fy.capture();
b_fz.capture();
b_xd.capture();
b_yd.capture();
b_zd.capture();
b_xdd.capture();
b_ydd.capture();
b_zdd.capture();
b_nodelist.capture();
b_determ.capture();
b_fx_elem.capture();
b_fy_elem.capture();
b_fz_elem.capture();
b_dvdx.capture();
b_dvdy.capture();
b_dvdz.capture();
b_x8n.capture();
b_y8n.capture();
b_z8n.capture();
b_sigxx.capture();
b_sigyy.capture();
b_sigzz.capture();
b_delv_xi.capture();
b_delx_xi.capture();
b_delv_eta.capture();
b_delx_eta.capture();
b_delv_zeta.capture();
b_delx_zeta.capture();
b_p.capture();
b_q.capture();
b_volo.capture();
b_v.capture();
b_vol_error.capture();
b_nodeElemStart.capture();
b_nodeElemCornerList.capture();
b_gamma.capture();
b_ss.capture();
b_elemMass.capture();
b_nodalMass.capture();
b_symmX.capture();
b_symmY.capture();
b_symmZ.capture();
b_vdov.capture();
b_delv.capture();
b_arealg.capture();
b_dxx.capture();
b_dyy.capture();
b_dzz.capture();
b_vnew.capture();
b_lzetam.capture();
b_lzetap.capture();
b_letap.capture();
b_letam.capture();
b_lxip.capture();
b_lxim.capture();
b_elemBC.capture();
b_ql.capture();
b_qq.capture();
b_e.capture();
b_elemRep.capture();
b_elemElem.capture();
if(initial){for(double x:params)result.push_back(x);for(int x:regions)result.push_back(x);}else{result.push_back(d.time());result.push_back(d.deltatime());result.push_back(d.cycle());result.push_back(d.dtcourant());result.push_back(d.dthydro());}}
void run(bool capture_output){for(int it=0;it<steps;it++){auto begin=std::chrono::steady_clock::now();TimeIncrement(d);Domain&domain=d;int numElem=d.numElem(),numNode=d.numNode(),numNodeBC=(nx+1)*(nx+1);double deltaTime=d.deltatime(),u_cut=d.u_cut(),hgcoef=d.hgcoef();int s1=d.symmXempty(),s2=d.symmYempty(),s3=d.symmZempty();
double monoq_limiter_mult=d.monoq_limiter_mult();
double monoq_max_slope=d.monoq_max_slope();
double qlc_monoq=d.qlc_monoq();
double qqc_monoq=d.qqc_monoq();
double e_cut=d.e_cut();
double p_cut=d.p_cut();
double q_cut=d.q_cut();
double v_cut=d.v_cut();
double eosvmax=d.eosvmax();
double eosvmin=d.eosvmin();
double pmin=d.pmin();
double emin=d.emin();
double rho0=d.refdens();
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)fill_sig(d_sigxx,d_sigyy,d_sigzz,d_p,d_q,numElem);
#else
fill_sig<<<(numElem+255)/256,256>>>(d_sigxx,d_sigyy,d_sigzz,d_p,d_q,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)integrateStress(d_fx_elem,d_fy_elem,d_fz_elem,d_x,d_y,d_z,d_nodelist,d_sigxx,d_sigyy,d_sigzz,d_determ,numElem);
#else
integrateStress<<<(numElem+255)/256,256>>>(d_fx_elem,d_fy_elem,d_fz_elem,d_x,d_y,d_z,d_nodelist,d_sigxx,d_sigyy,d_sigzz,d_determ,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNode;cpu_gidx++)acc_final_force(d_fx_elem,d_fy_elem,d_fz_elem,d_fx,d_fy,d_fz,d_nodeElemStart,d_nodeElemCornerList,numNode);
#else
acc_final_force<<<(numNode+255)/256,256>>>(d_fx_elem,d_fy_elem,d_fz_elem,d_fx,d_fy,d_fz,d_nodeElemStart,d_nodeElemCornerList,numNode);ck(cudaGetLastError());
#endif
int ve=-1;
#ifdef CPU_REFERENCE
d_vol_error[0]=ve;
#else
ck(cudaMemcpy(d_vol_error,&ve,4,cudaMemcpyHostToDevice));
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)hgc(d_dvdx,d_dvdy,d_dvdz,d_x8n,d_y8n,d_z8n,d_determ,d_x,d_y,d_z,d_nodelist,d_volo,d_v,d_vol_error,numElem);
#else
hgc<<<(numElem+255)/256,256>>>(d_dvdx,d_dvdy,d_dvdz,d_x8n,d_y8n,d_z8n,d_determ,d_x,d_y,d_z,d_nodelist,d_volo,d_v,d_vol_error,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)fb(d_dvdx,d_dvdy,d_dvdz,d_x8n,d_y8n,d_z8n,d_determ,d_xd,d_yd,d_zd,d_ss,d_elemMass,d_nodelist,d_gamma,d_fx_elem,d_fy_elem,d_fz_elem,hgcoef,numElem);
#else
fb<<<(numElem+255)/256,256>>>(d_dvdx,d_dvdy,d_dvdz,d_x8n,d_y8n,d_z8n,d_determ,d_xd,d_yd,d_zd,d_ss,d_elemMass,d_nodelist,d_gamma,d_fx_elem,d_fy_elem,d_fz_elem,hgcoef,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNode;cpu_gidx++)collect_final_force(d_fx_elem,d_fy_elem,d_fz_elem,d_fx,d_fy,d_fz,d_nodeElemStart,d_nodeElemCornerList,numNode);
#else
collect_final_force<<<(numNode+255)/256,256>>>(d_fx_elem,d_fy_elem,d_fz_elem,d_fx,d_fy,d_fz,d_nodeElemStart,d_nodeElemCornerList,numNode);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNode;cpu_gidx++)accelerationForNode(d_fx,d_fy,d_fz,d_nodalMass,d_xdd,d_ydd,d_zdd,numNode);
#else
accelerationForNode<<<(numNode+255)/256,256>>>(d_fx,d_fy,d_fz,d_nodalMass,d_xdd,d_ydd,d_zdd,numNode);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNodeBC;cpu_gidx++)applyAccelerationBoundaryConditionsForNodes(d_symmX,d_symmY,d_symmZ,d_xdd,d_ydd,d_zdd,s1,s2,s3,numNodeBC);
#else
applyAccelerationBoundaryConditionsForNodes<<<(numNodeBC+255)/256,256>>>(d_symmX,d_symmY,d_symmZ,d_xdd,d_ydd,d_zdd,s1,s2,s3,numNodeBC);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNode;cpu_gidx++)calcVelocityForNodes(d_xd,d_yd,d_zd,d_xdd,d_ydd,d_zdd,deltaTime,u_cut,numNode);
#else
calcVelocityForNodes<<<(numNode+255)/256,256>>>(d_xd,d_yd,d_zd,d_xdd,d_ydd,d_zdd,deltaTime,u_cut,numNode);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numNode;cpu_gidx++)calcPositionForNodes(d_x,d_y,d_z,d_xd,d_yd,d_zd,deltaTime,numNode);
#else
calcPositionForNodes<<<(numNode+255)/256,256>>>(d_x,d_y,d_z,d_xd,d_yd,d_zd,deltaTime,numNode);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)calcKinematicsForElems(d_xd,d_yd,d_zd,d_x,d_y,d_z,d_nodelist,d_volo,d_v,d_delv,d_arealg,d_dxx,d_dyy,d_dzz,d_vnew,deltaTime,numElem);
#else
calcKinematicsForElems<<<(numElem+255)/256,256>>>(d_xd,d_yd,d_zd,d_x,d_y,d_z,d_nodelist,d_volo,d_v,d_delv,d_arealg,d_dxx,d_dyy,d_dzz,d_vnew,deltaTime,numElem);ck(cudaGetLastError());
#endif
ve=-1;
#ifdef CPU_REFERENCE
d_vol_error[0]=ve;
#else
ck(cudaMemcpy(d_vol_error,&ve,4,cudaMemcpyHostToDevice));
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)calcStrainRates(d_dxx,d_dyy,d_dzz,d_vnew,d_vdov,d_vol_error,numElem);
#else
calcStrainRates<<<(numElem+255)/256,256>>>(d_dxx,d_dyy,d_dzz,d_vnew,d_vdov,d_vol_error,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
std::copy(d_vdov,d_vdov+numElem,d.m_vdov.begin());
#else
ck(cudaMemcpy(d.m_vdov.data(),d_vdov,8*numElem,cudaMemcpyDeviceToHost));
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)calcMonotonicQGradientsForElems(d_xd,d_yd,d_zd,d_x,d_y,d_z,d_nodelist,d_volo,d_delv_eta,d_delx_eta,d_delv_zeta,d_delx_zeta,d_delv_xi,d_delx_xi,d_vnew,numElem);
#else
calcMonotonicQGradientsForElems<<<(numElem+255)/256,256>>>(d_xd,d_yd,d_zd,d_x,d_y,d_z,d_nodelist,d_volo,d_delv_eta,d_delx_eta,d_delv_zeta,d_delx_zeta,d_delv_xi,d_delx_xi,d_vnew,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)calcMonotonicQForElems(d_elemBC,d_elemMass,d_ql,d_qq,d_vdov,d_volo,d_delv_eta,d_delx_eta,d_delv_zeta,d_delx_zeta,d_delv_xi,d_delx_xi,d_lxim,d_lxip,d_lzetam,d_lzetap,d_letap,d_letam,d_vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem);
#else
calcMonotonicQForElems<<<(numElem+255)/256,256>>>(d_elemBC,d_elemMass,d_ql,d_qq,d_vdov,d_volo,d_delv_eta,d_delx_eta,d_delv_zeta,d_delx_zeta,d_delv_xi,d_delx_xi,d_lxim,d_lxip,d_lzetam,d_lzetap,d_letap,d_letam,d_vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
for(cpu_gidx=0;cpu_gidx<numElem;cpu_gidx++)applyMaterialPropertiesForElems(d_ql,d_qq,d_delv,d_elemRep,d_elemElem,d_q,d_p,d_e,d_ss,d_v,d_vnew,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem);
#else
applyMaterialPropertiesForElems<<<(numElem+255)/256,256>>>(d_ql,d_qq,d_delv,d_elemRep,d_elemElem,d_q,d_p,d_e,d_ss,d_v,d_vnew,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem);ck(cudaGetLastError());
#endif
#ifdef CPU_REFERENCE
std::copy(d_arealg,d_arealg+numElem,d.m_arealg.begin());std::copy(d_ss,d_ss+numElem,d.m_ss.begin());
#else
ck(cudaMemcpy(d.m_arealg.data(),d_arealg,8*numElem,cudaMemcpyDeviceToHost));ck(cudaMemcpy(d.m_ss.data(),d_ss,8*numElem,cudaMemcpyDeviceToHost));
#endif
CalcTimeConstraintsForElems(d);
#ifndef CPU_REFERENCE
ck(cudaDeviceSynchronize());
#endif
seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();if(capture_output)capture(false);}}
};
int main(){std::vector<State*>states;
states.push_back(new State(0,1,1,0));
states.push_back(new State(1,3,4,0));
states.push_back(new State(2,7,4,0));
states.push_back(new State(3,30,3,0));
states.push_back(new State(4,3,1,1));
states.push_back(new State(5,3,1,2));
states.push_back(new State(6,3,1,3));
for(auto*s:states)s->run(false);for(int trial=1;trial<=5;trial++){result.clear();for(auto*s:states){s->reset();s->capture(true);}double sec=0;for(auto*s:states){s->run(true);sec+=s->seconds;}
#ifdef CPU_REFERENCE
FILE*f=fopen("source_cpu_oracle_f64.bin","wb");fwrite(result.data(),8,result.size(),f);fclose(f);printf("CPU fullstate words %zu\n",result.size());
#else
FILE*f=fopen("source_cpu_oracle_f64.bin","rb");if(!f)exit(4);std::vector<double>oracle(result.size());if(fread(oracle.data(),8,oracle.size(),f)!=oracle.size())exit(5);fclose(f);bool pass=true;double maxerr=0;for(size_t i=0;i<result.size();i++){double e=fabs(result[i]-oracle[i]);maxerr=std::max(e,maxerr);if(!std::isfinite(result[i])||e>1e-7+3e-8*fabs(oracle[i])){if(pass)fprintf(stderr,"FIRST mismatch %zu %.17g %.17g\n",i,result[i],oracle[i]);pass=false;}}std::string path="lulesh_trial"+std::to_string(trial)+".bin";f=fopen(path.c_str(),"wb");fwrite(result.data(),8,result.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"time_seconds\":%.12g,\"max_error\":%.12g,\"timing_region\":\"lulesh_full15kernel_host_adaptive_cycles_excluding_allocation_capture\",\"jit_excluded\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu}}\n",trial,pass?"true":"false",sec,maxerr,path.c_str(),result.size());fflush(stdout);
#endif
}
for(auto*s:states)delete s;}
