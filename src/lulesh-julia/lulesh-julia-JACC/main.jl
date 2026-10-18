using CUDA
import JACC
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function fill_sig_kernel(i,sigxx,sigyy,sigzz,p,q,numElem)
 fill_sig_body!(sigxx,sigyy,sigzz,p,q,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:fill_sig},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,fill_sig_kernel,args...)
 nothing
end
function integrateStress_kernel(i,fx_elem,fy_elem,fz_elem,x,y,z,nodelist,sigxx,sigyy,sigzz,determ,numElem)
 integrateStress_body!(fx_elem,fy_elem,fz_elem,x,y,z,nodelist,sigxx,sigyy,sigzz,determ,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:integrateStress},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,integrateStress_kernel,args...)
 nothing
end
function acc_final_force_kernel(i,fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode)
 acc_final_force_body!(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode,Int32(i-1))
 nothing
end
function launch(::Val{:acc_final_force},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,acc_final_force_kernel,args...)
 nothing
end
function hgc_kernel(i,dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,x,y,z,nodelist,volo,v,vol_error,numElem)
 hgc_body!(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,x,y,z,nodelist,volo,v,vol_error,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:hgc},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,hgc_kernel,args...)
 nothing
end
function fb_kernel(i,dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,xd,yd,zd,ss,elemMass,nodelist,gamma,fx_elem,fy_elem,fz_elem,hgcoef,numElem)
 fb_body!(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,xd,yd,zd,ss,elemMass,nodelist,gamma,fx_elem,fy_elem,fz_elem,hgcoef,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:fb},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,fb_kernel,args...)
 nothing
end
function collect_final_force_kernel(i,fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode)
 collect_final_force_body!(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode,Int32(i-1))
 nothing
end
function launch(::Val{:collect_final_force},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,collect_final_force_kernel,args...)
 nothing
end
function accelerationForNode_kernel(i,fx,fy,fz,nodalMass,xdd,ydd,zdd,numNode)
 accelerationForNode_body!(fx,fy,fz,nodalMass,xdd,ydd,zdd,numNode,Int32(i-1))
 nothing
end
function launch(::Val{:accelerationForNode},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,accelerationForNode_kernel,args...)
 nothing
end
function applyAccelerationBoundaryConditionsForNodes_kernel(i,symmX,symmY,symmZ,xdd,ydd,zdd,s1,s2,s3,numNodeBC)
 applyAccelerationBoundaryConditionsForNodes_body!(symmX,symmY,symmZ,xdd,ydd,zdd,s1,s2,s3,numNodeBC,Int32(i-1))
 nothing
end
function launch(::Val{:applyAccelerationBoundaryConditionsForNodes},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,applyAccelerationBoundaryConditionsForNodes_kernel,args...)
 nothing
end
function calcVelocityForNodes_kernel(i,xd,yd,zd,xdd,ydd,zdd,deltaTime,u_cut,numNode)
 calcVelocityForNodes_body!(xd,yd,zd,xdd,ydd,zdd,deltaTime,u_cut,numNode,Int32(i-1))
 nothing
end
function launch(::Val{:calcVelocityForNodes},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcVelocityForNodes_kernel,args...)
 nothing
end
function calcPositionForNodes_kernel(i,x,y,z,xd,yd,zd,deltaTime,numNode)
 calcPositionForNodes_body!(x,y,z,xd,yd,zd,deltaTime,numNode,Int32(i-1))
 nothing
end
function launch(::Val{:calcPositionForNodes},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcPositionForNodes_kernel,args...)
 nothing
end
function calcKinematicsForElems_kernel(i,xd,yd,zd,x,y,z,nodeList,volo,v,delv,arealg,dxx,dyy,dzz,vnew,deltaTime,numElem)
 calcKinematicsForElems_body!(xd,yd,zd,x,y,z,nodeList,volo,v,delv,arealg,dxx,dyy,dzz,vnew,deltaTime,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:calcKinematicsForElems},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcKinematicsForElems_kernel,args...)
 nothing
end
function calcStrainRates_kernel(i,dxx,dyy,dzz,vnew,vdov,vol_error,numElem)
 calcStrainRates_body!(dxx,dyy,dzz,vnew,vdov,vol_error,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:calcStrainRates},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcStrainRates_kernel,args...)
 nothing
end
function calcMonotonicQGradientsForElems_kernel(i,xd,yd,zd,x,y,z,nodelist,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,vnew,numElem)
 calcMonotonicQGradientsForElems_body!(xd,yd,zd,x,y,z,nodelist,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,vnew,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:calcMonotonicQGradientsForElems},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcMonotonicQGradientsForElems_kernel,args...)
 nothing
end
function calcMonotonicQForElems_kernel(i,elemBC,elemMass,ql,qq,vdov,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,lxim,lxip,lzetam,lzetap,letap,letam,vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem)
 calcMonotonicQForElems_body!(elemBC,elemMass,ql,qq,vdov,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,lxim,lxip,lzetam,lzetap,letap,letam,vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:calcMonotonicQForElems},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,calcMonotonicQForElems_kernel,args...)
 nothing
end
function applyMaterialPropertiesForElems_kernel(i,ql,qq,delv,elemRep,elemElem,q,p,e,ss,v,vnewc,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem)
 applyMaterialPropertiesForElems_body!(ql,qq,delv,elemRep,elemElem,q,p,e,ss,v,vnewc,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem,Int32(i-1))
 nothing
end
function launch(::Val{:applyMaterialPropertiesForElems},n,args...)
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,applyMaterialPropertiesForElems_kernel,args...)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
