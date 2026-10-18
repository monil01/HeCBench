using CUDA
using KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function fill_sig_kernel(sigxx,sigyy,sigzz,p,q,numElem)
 i=@index(Global,Linear)
 fill_sig_body!(sigxx,sigyy,sigzz,p,q,numElem,Int32(i-1))
end
function launch(::Val{:fill_sig},n,args...)
 fill_sig_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function integrateStress_kernel(fx_elem,fy_elem,fz_elem,x,y,z,nodelist,sigxx,sigyy,sigzz,determ,numElem)
 i=@index(Global,Linear)
 integrateStress_body!(fx_elem,fy_elem,fz_elem,x,y,z,nodelist,sigxx,sigyy,sigzz,determ,numElem,Int32(i-1))
end
function launch(::Val{:integrateStress},n,args...)
 integrateStress_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function acc_final_force_kernel(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode)
 i=@index(Global,Linear)
 acc_final_force_body!(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode,Int32(i-1))
end
function launch(::Val{:acc_final_force},n,args...)
 acc_final_force_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function hgc_kernel(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,x,y,z,nodelist,volo,v,vol_error,numElem)
 i=@index(Global,Linear)
 hgc_body!(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,x,y,z,nodelist,volo,v,vol_error,numElem,Int32(i-1))
end
function launch(::Val{:hgc},n,args...)
 hgc_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function fb_kernel(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,xd,yd,zd,ss,elemMass,nodelist,gamma,fx_elem,fy_elem,fz_elem,hgcoef,numElem)
 i=@index(Global,Linear)
 fb_body!(dvdx,dvdy,dvdz,x8n,y8n,z8n,determ,xd,yd,zd,ss,elemMass,nodelist,gamma,fx_elem,fy_elem,fz_elem,hgcoef,numElem,Int32(i-1))
end
function launch(::Val{:fb},n,args...)
 fb_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function collect_final_force_kernel(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode)
 i=@index(Global,Linear)
 collect_final_force_body!(fx_elem,fy_elem,fz_elem,fx,fy,fz,nodeElemStart,nodeElemCornerList,numNode,Int32(i-1))
end
function launch(::Val{:collect_final_force},n,args...)
 collect_final_force_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function accelerationForNode_kernel(fx,fy,fz,nodalMass,xdd,ydd,zdd,numNode)
 i=@index(Global,Linear)
 accelerationForNode_body!(fx,fy,fz,nodalMass,xdd,ydd,zdd,numNode,Int32(i-1))
end
function launch(::Val{:accelerationForNode},n,args...)
 accelerationForNode_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function applyAccelerationBoundaryConditionsForNodes_kernel(symmX,symmY,symmZ,xdd,ydd,zdd,s1,s2,s3,numNodeBC)
 i=@index(Global,Linear)
 applyAccelerationBoundaryConditionsForNodes_body!(symmX,symmY,symmZ,xdd,ydd,zdd,s1,s2,s3,numNodeBC,Int32(i-1))
end
function launch(::Val{:applyAccelerationBoundaryConditionsForNodes},n,args...)
 applyAccelerationBoundaryConditionsForNodes_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcVelocityForNodes_kernel(xd,yd,zd,xdd,ydd,zdd,deltaTime,u_cut,numNode)
 i=@index(Global,Linear)
 calcVelocityForNodes_body!(xd,yd,zd,xdd,ydd,zdd,deltaTime,u_cut,numNode,Int32(i-1))
end
function launch(::Val{:calcVelocityForNodes},n,args...)
 calcVelocityForNodes_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcPositionForNodes_kernel(x,y,z,xd,yd,zd,deltaTime,numNode)
 i=@index(Global,Linear)
 calcPositionForNodes_body!(x,y,z,xd,yd,zd,deltaTime,numNode,Int32(i-1))
end
function launch(::Val{:calcPositionForNodes},n,args...)
 calcPositionForNodes_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcKinematicsForElems_kernel(xd,yd,zd,x,y,z,nodeList,volo,v,delv,arealg,dxx,dyy,dzz,vnew,deltaTime,numElem)
 i=@index(Global,Linear)
 calcKinematicsForElems_body!(xd,yd,zd,x,y,z,nodeList,volo,v,delv,arealg,dxx,dyy,dzz,vnew,deltaTime,numElem,Int32(i-1))
end
function launch(::Val{:calcKinematicsForElems},n,args...)
 calcKinematicsForElems_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcStrainRates_kernel(dxx,dyy,dzz,vnew,vdov,vol_error,numElem)
 i=@index(Global,Linear)
 calcStrainRates_body!(dxx,dyy,dzz,vnew,vdov,vol_error,numElem,Int32(i-1))
end
function launch(::Val{:calcStrainRates},n,args...)
 calcStrainRates_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcMonotonicQGradientsForElems_kernel(xd,yd,zd,x,y,z,nodelist,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,vnew,numElem)
 i=@index(Global,Linear)
 calcMonotonicQGradientsForElems_body!(xd,yd,zd,x,y,z,nodelist,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,vnew,numElem,Int32(i-1))
end
function launch(::Val{:calcMonotonicQGradientsForElems},n,args...)
 calcMonotonicQGradientsForElems_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function calcMonotonicQForElems_kernel(elemBC,elemMass,ql,qq,vdov,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,lxim,lxip,lzetam,lzetap,letap,letam,vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem)
 i=@index(Global,Linear)
 calcMonotonicQForElems_body!(elemBC,elemMass,ql,qq,vdov,volo,delv_eta,delx_eta,delv_zeta,delx_zeta,delv_xi,delx_xi,lxim,lxip,lzetam,lzetap,letap,letam,vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem,Int32(i-1))
end
function launch(::Val{:calcMonotonicQForElems},n,args...)
 calcMonotonicQForElems_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
@kernel function applyMaterialPropertiesForElems_kernel(ql,qq,delv,elemRep,elemElem,q,p,e,ss,v,vnewc,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem)
 i=@index(Global,Linear)
 applyMaterialPropertiesForElems_body!(ql,qq,delv,elemRep,elemElem,q,p,e,ss,v,vnewc,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem,Int32(i-1))
end
function launch(::Val{:applyMaterialPropertiesForElems},n,args...)
 applyMaterialPropertiesForElems_kernel(CUDABackend(),256)(args...;ndrange=cld(n,256)*256)
 nothing
end
allocate(h)=begin a=KernelAbstractions.allocate(CUDABackend(),eltype(h),size(h));copyto!(a,h);a;end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
