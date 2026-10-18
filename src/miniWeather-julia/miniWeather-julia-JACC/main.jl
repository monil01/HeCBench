using CUDA
import JACC
import Atomix
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function zhalo!(xx,yy,state,nx::Int32,nz::Int32,hs::Int32)
 i=Int32(xx)-Int32(1)
 ll=Int32(yy)-Int32(1)
 update_state_z_body!(state,nx,nz,hs,i,ll)
 return nothing
end
function fx!(xx,yy,state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 i=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 compute_flux_x_body!(state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef,nx,nz,hs,i,k)
 return nothing
end
function tx!(xx,yy,zz,flux,tend,nx::Int32,nz::Int32,dx::Int32)
 i=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 ll=Int32(zz)-Int32(1)
 compute_tend_x_body!(flux,tend,nx,nz,dx,i,k,ll)
 return nothing
end
function fz!(xx,yy,state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 i=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 compute_flux_z_body!(state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef,nx,nz,hs,i,k)
 return nothing
end
function tz!(xx,yy,zz,state,flux,tend,nx::Int32,nz::Int32,hs::Int32,dz::Int32)
 i=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 ll=Int32(zz)-Int32(1)
 compute_tend_z_body!(state,flux,tend,nx,nz,hs,dz,i,k,ll)
 return nothing
end
function update!(xx,yy,zz,state_init,state_out,tend,nx::Int32,nz::Int32,hs::Int32,dt::Float64)
 i=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 ll=Int32(zz)-Int32(1)
 update_fluid_state_body!(state_init,state_out,tend,nx,nz,hs,dt,i,k,ll)
 return nothing
end
function pack!(xx,yy,zz,state,sl,sr,nx::Int32,nz::Int32,hs::Int32)
 s=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 ll=Int32(zz)-Int32(1)
 pack_body!(state,sl,sr,nx,nz,hs,s,k,ll)
 return nothing
end
function unpack!(xx,yy,zz,state,rl,rr,nx::Int32,nz::Int32,hs::Int32)
 s=Int32(xx)-Int32(1)
 k=Int32(yy)-Int32(1)
 ll=Int32(zz)-Int32(1)
 unpack_body!(state,rl,rr,nx,nz,hs,s,k,ll)
 return nothing
end
function mass!(xx,yy,mass,te,state,h0,h1,nx::Int32,nz::Int32,dx::Float64,dz::Float64)
 i=Int32(xx)-Int32(1);k=Int32(yy)-Int32(1)
 if i<nx && k<nz
  m,e=mass_terms_body(state,h0,h1,nx,nz,dx,dz,i,k)
  Atomix.@atomic mass[1]+=m
  Atomix.@atomic te[1]+=e
 end
 return nothing
end
function launch(which,args,blocks,threads)
 spec=JACC.launch_spec(;threads=threads,blocks=blocks,shmem_size=0,sync=false)
 if which==:zhalo
  JACC.parallel_for(spec,threads .* blocks,zhalo!,args...)
 elseif which==:fx
  JACC.parallel_for(spec,threads .* blocks,fx!,args...)
 elseif which==:tx
  JACC.parallel_for(spec,threads .* blocks,tx!,args...)
 elseif which==:fz
  JACC.parallel_for(spec,threads .* blocks,fz!,args...)
 elseif which==:tz
  JACC.parallel_for(spec,threads .* blocks,tz!,args...)
 elseif which==:update
  JACC.parallel_for(spec,threads .* blocks,update!,args...)
 elseif which==:pack
  JACC.parallel_for(spec,threads .* blocks,pack!,args...)
 elseif which==:unpack
  JACC.parallel_for(spec,threads .* blocks,unpack!,args...)
 elseif which==:mass
  JACC.parallel_for(spec,threads .* blocks,mass!,args...)
 else;error("Unknown sourcekernel");end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
