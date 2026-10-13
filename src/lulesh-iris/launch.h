#pragma once
#include <iris/iris.hpp>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
struct Buffer {
 std::vector<unsigned char> host;
 iris::DMem mem;
 explicit Buffer(size_t bytes):host(bytes),mem(host.data(),bytes){}
 void update(const void *src,size_t bytes){if(bytes>host.size())exit(4);std::memcpy(host.data(),src,bytes);mem.update(host.data());}
 void read(void *dst,size_t bytes){if(bytes>host.size())exit(4);iris::Task task;task.flush_out(mem);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);std::memcpy(dst,host.data(),bytes);}
};
struct Dim3 {size_t x;Dim3(size_t v):x(v){}};
inline void launch_fill_sig(Dim3 grid,Dim3 block,Buffer *sigxx,Buffer *sigyy,Buffer *sigzz,Buffer *p,Buffer *q,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&sigxx->mem,&sigyy->mem,&sigzz->mem,&p->mem,&q->mem,const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_r,sizeof(numElem)};
 task.kernel("fill_sig",1,nullptr,g,l,6,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_integrateStress(Dim3 grid,Dim3 block,Buffer *fx_elem,Buffer *fy_elem,Buffer *fz_elem,Buffer *x,Buffer *y,Buffer *z,Buffer *nodelist,Buffer *sigxx,Buffer *sigyy,Buffer *sigzz,Buffer *determ,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&fx_elem->mem,&fy_elem->mem,&fz_elem->mem,&x->mem,&y->mem,&z->mem,&nodelist->mem,&sigxx->mem,&sigyy->mem,&sigzz->mem,&determ->mem,const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(numElem)};
 task.kernel("integrateStress",1,nullptr,g,l,12,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_acc_final_force(Dim3 grid,Dim3 block,Buffer *fx_elem,Buffer *fy_elem,Buffer *fz_elem,Buffer *fx,Buffer *fy,Buffer *fz,Buffer *nodeElemStart,Buffer *nodeElemCornerList,const Index_t numNode) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&fx_elem->mem,&fy_elem->mem,&fz_elem->mem,&fx->mem,&fy->mem,&fz->mem,&nodeElemStart->mem,&nodeElemCornerList->mem,const_cast<void *>(static_cast<const void *>(&numNode))};
 int info[]={iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_r,iris_r,sizeof(numNode)};
 task.kernel("acc_final_force",1,nullptr,g,l,9,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_hgc(Dim3 grid,Dim3 block,Buffer *dvdx,Buffer *dvdy,Buffer *dvdz,Buffer *x8n,Buffer *y8n,Buffer *z8n,Buffer *determ,Buffer *x,Buffer *y,Buffer *z,Buffer *nodelist,Buffer *volo,Buffer *v,Buffer *vol_error,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&dvdx->mem,&dvdy->mem,&dvdz->mem,&x8n->mem,&y8n->mem,&z8n->mem,&determ->mem,&x->mem,&y->mem,&z->mem,&nodelist->mem,&volo->mem,&v->mem,&vol_error->mem,const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(numElem)};
 task.kernel("hgc",1,nullptr,g,l,15,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_fb(Dim3 grid,Dim3 block,Buffer *dvdx,Buffer *dvdy,Buffer *dvdz,Buffer *x8n,Buffer *y8n,Buffer *z8n,Buffer *determ,Buffer *xd,Buffer *yd,Buffer *zd,Buffer *ss,Buffer *elemMass,Buffer *nodelist,Buffer *gamma,Buffer *fx_elem,Buffer *fy_elem,Buffer *fz_elem,Real_t hgcoef,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&dvdx->mem,&dvdy->mem,&dvdz->mem,&x8n->mem,&y8n->mem,&z8n->mem,&determ->mem,&xd->mem,&yd->mem,&zd->mem,&ss->mem,&elemMass->mem,&nodelist->mem,&gamma->mem,&fx_elem->mem,&fy_elem->mem,&fz_elem->mem,const_cast<void *>(static_cast<const void *>(&hgcoef)),const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,sizeof(hgcoef),sizeof(numElem)};
 task.kernel("fb",1,nullptr,g,l,19,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_collect_final_force(Dim3 grid,Dim3 block,Buffer *fx_elem,Buffer *fy_elem,Buffer *fz_elem,Buffer *fx,Buffer *fy,Buffer *fz,Buffer *nodeElemStart,Buffer *nodeElemCornerList,const Index_t numNode) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&fx_elem->mem,&fy_elem->mem,&fz_elem->mem,&fx->mem,&fy->mem,&fz->mem,&nodeElemStart->mem,&nodeElemCornerList->mem,const_cast<void *>(static_cast<const void *>(&numNode))};
 int info[]={iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_r,iris_r,sizeof(numNode)};
 task.kernel("collect_final_force",1,nullptr,g,l,9,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_accelerationForNode(Dim3 grid,Dim3 block,Buffer *fx,Buffer *fy,Buffer *fz,Buffer *nodalMass,Buffer *xdd,Buffer *ydd,Buffer *zdd,const Index_t numNode) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&fx->mem,&fy->mem,&fz->mem,&nodalMass->mem,&xdd->mem,&ydd->mem,&zdd->mem,const_cast<void *>(static_cast<const void *>(&numNode))};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,sizeof(numNode)};
 task.kernel("accelerationForNode",1,nullptr,g,l,8,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_applyAccelerationBoundaryConditionsForNodes(Dim3 grid,Dim3 block,Buffer *symmX,Buffer *symmY,Buffer *symmZ,Buffer *xdd,Buffer *ydd,Buffer *zdd,const Index_t s1,const Index_t s2,const Index_t s3,const Index_t numNodeBC) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&symmX->mem,&symmY->mem,&symmZ->mem,&xdd->mem,&ydd->mem,&zdd->mem,const_cast<void *>(static_cast<const void *>(&s1)),const_cast<void *>(static_cast<const void *>(&s2)),const_cast<void *>(static_cast<const void *>(&s3)),const_cast<void *>(static_cast<const void *>(&numNodeBC))};
 int info[]={iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,sizeof(s1),sizeof(s2),sizeof(s3),sizeof(numNodeBC)};
 task.kernel("applyAccelerationBoundaryConditionsForNodes",1,nullptr,g,l,10,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcVelocityForNodes(Dim3 grid,Dim3 block,Buffer *xd,Buffer *yd,Buffer *zd,Buffer *xdd,Buffer *ydd,Buffer *zdd,const Real_t deltaTime,const Real_t u_cut,const Index_t numNode) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&xd->mem,&yd->mem,&zd->mem,&xdd->mem,&ydd->mem,&zdd->mem,const_cast<void *>(static_cast<const void *>(&deltaTime)),const_cast<void *>(static_cast<const void *>(&u_cut)),const_cast<void *>(static_cast<const void *>(&numNode))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,sizeof(deltaTime),sizeof(u_cut),sizeof(numNode)};
 task.kernel("calcVelocityForNodes",1,nullptr,g,l,9,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcPositionForNodes(Dim3 grid,Dim3 block,Buffer *x,Buffer *y,Buffer *z,Buffer *xd,Buffer *yd,Buffer *zd,const Real_t deltaTime,const Index_t numNode) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&x->mem,&y->mem,&z->mem,&xd->mem,&yd->mem,&zd->mem,const_cast<void *>(static_cast<const void *>(&deltaTime)),const_cast<void *>(static_cast<const void *>(&numNode))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,sizeof(deltaTime),sizeof(numNode)};
 task.kernel("calcPositionForNodes",1,nullptr,g,l,8,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcKinematicsForElems(Dim3 grid,Dim3 block,Buffer *xd,Buffer *yd,Buffer *zd,Buffer *x,Buffer *y,Buffer *z,Buffer *nodeList,Buffer *volo,Buffer *v,Buffer *delv,Buffer *arealg,Buffer *dxx,Buffer *dyy,Buffer *dzz,Buffer *vnew,const Real_t deltaTime,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&xd->mem,&yd->mem,&zd->mem,&x->mem,&y->mem,&z->mem,&nodeList->mem,&volo->mem,&v->mem,&delv->mem,&arealg->mem,&dxx->mem,&dyy->mem,&dzz->mem,&vnew->mem,const_cast<void *>(static_cast<const void *>(&deltaTime)),const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(deltaTime),sizeof(numElem)};
 task.kernel("calcKinematicsForElems",1,nullptr,g,l,17,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcStrainRates(Dim3 grid,Dim3 block,Buffer *dxx,Buffer *dyy,Buffer *dzz,Buffer *vnew,Buffer *vdov,Buffer *vol_error,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&dxx->mem,&dyy->mem,&dzz->mem,&vnew->mem,&vdov->mem,&vol_error->mem,const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_rw,iris_rw,iris_rw,iris_r,iris_rw,iris_rw,sizeof(numElem)};
 task.kernel("calcStrainRates",1,nullptr,g,l,7,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcMonotonicQGradientsForElems(Dim3 grid,Dim3 block,Buffer *xd,Buffer *yd,Buffer *zd,Buffer *x,Buffer *y,Buffer *z,Buffer *nodelist,Buffer *volo,Buffer *delv_eta,Buffer *delx_eta,Buffer *delv_zeta,Buffer *delx_zeta,Buffer *delv_xi,Buffer *delx_xi,Buffer *vnew,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&xd->mem,&yd->mem,&zd->mem,&x->mem,&y->mem,&z->mem,&nodelist->mem,&volo->mem,&delv_eta->mem,&delx_eta->mem,&delv_zeta->mem,&delx_zeta->mem,&delv_xi->mem,&delx_xi->mem,&vnew->mem,const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_r,sizeof(numElem)};
 task.kernel("calcMonotonicQGradientsForElems",1,nullptr,g,l,16,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_calcMonotonicQForElems(Dim3 grid,Dim3 block,Buffer *elemBC,Buffer *elemMass,Buffer *ql,Buffer *qq,Buffer *vdov,Buffer *volo,Buffer *delv_eta,Buffer *delx_eta,Buffer *delv_zeta,Buffer *delx_zeta,Buffer *delv_xi,Buffer *delx_xi,Buffer *lxim,Buffer *lxip,Buffer *lzetam,Buffer *lzetap,Buffer *letap,Buffer *letam,Buffer *vnew,const Real_t monoq_limiter_mult,const Real_t monoq_max_slope,const Real_t qlc_monoq,const Real_t qqc_monoq,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&elemBC->mem,&elemMass->mem,&ql->mem,&qq->mem,&vdov->mem,&volo->mem,&delv_eta->mem,&delx_eta->mem,&delv_zeta->mem,&delx_zeta->mem,&delv_xi->mem,&delx_xi->mem,&lxim->mem,&lxip->mem,&lzetam->mem,&lzetap->mem,&letap->mem,&letam->mem,&vnew->mem,const_cast<void *>(static_cast<const void *>(&monoq_limiter_mult)),const_cast<void *>(static_cast<const void *>(&monoq_max_slope)),const_cast<void *>(static_cast<const void *>(&qlc_monoq)),const_cast<void *>(static_cast<const void *>(&qqc_monoq)),const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_r,iris_r,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,sizeof(monoq_limiter_mult),sizeof(monoq_max_slope),sizeof(qlc_monoq),sizeof(qqc_monoq),sizeof(numElem)};
 task.kernel("calcMonotonicQForElems",1,nullptr,g,l,24,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_applyMaterialPropertiesForElems(Dim3 grid,Dim3 block,Buffer *ql,Buffer *qq,Buffer *delv,Buffer *elemRep,Buffer *elemElem,Buffer *q,Buffer *p,Buffer *e,Buffer *ss,Buffer *v,Buffer *vnewc,const Real_t  e_cut,const Real_t  p_cut,const Real_t  q_cut,const Real_t  v_cut,const Real_t eosvmax,const Real_t eosvmin,const Real_t pmin,const Real_t emin,const Real_t rho0,const Index_t numElem) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&ql->mem,&qq->mem,&delv->mem,&elemRep->mem,&elemElem->mem,&q->mem,&p->mem,&e->mem,&ss->mem,&v->mem,&vnewc->mem,const_cast<void *>(static_cast<const void *>(&e_cut)),const_cast<void *>(static_cast<const void *>(&p_cut)),const_cast<void *>(static_cast<const void *>(&q_cut)),const_cast<void *>(static_cast<const void *>(&v_cut)),const_cast<void *>(static_cast<const void *>(&eosvmax)),const_cast<void *>(static_cast<const void *>(&eosvmin)),const_cast<void *>(static_cast<const void *>(&pmin)),const_cast<void *>(static_cast<const void *>(&emin)),const_cast<void *>(static_cast<const void *>(&rho0)),const_cast<void *>(static_cast<const void *>(&numElem))};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(e_cut),sizeof(p_cut),sizeof(q_cut),sizeof(v_cut),sizeof(eosvmax),sizeof(eosvmin),sizeof(pmin),sizeof(emin),sizeof(rho0),sizeof(numElem)};
 task.kernel("applyMaterialPropertiesForElems",1,nullptr,g,l,21,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
