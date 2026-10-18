using CUDA
using KernelAbstractions
import Atomix
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function zhalo!(state,nx::Int32,nz::Int32,hs::Int32)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 ll=Int32(positions[2])-Int32(1)
 update_state_z_body!(state,nx,nz,hs,i,ll)
end
@kernel function fx!(state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 compute_flux_x_body!(state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef,nx,nz,hs,i,k)
end
@kernel function tx!(flux,tend,nx::Int32,nz::Int32,dx::Int32)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 ll=Int32(positions[3])-Int32(1)
 compute_tend_x_body!(flux,tend,nx,nz,dx,i,k,ll)
end
@kernel function fz!(state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 compute_flux_z_body!(state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef,nx,nz,hs,i,k)
end
@kernel function tz!(state,flux,tend,nx::Int32,nz::Int32,hs::Int32,dz::Int32)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 ll=Int32(positions[3])-Int32(1)
 compute_tend_z_body!(state,flux,tend,nx,nz,hs,dz,i,k,ll)
end
@kernel function update!(state_init,state_out,tend,nx::Int32,nz::Int32,hs::Int32,dt::Float64)
 positions=@index(Global,NTuple)
 i=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 ll=Int32(positions[3])-Int32(1)
 update_fluid_state_body!(state_init,state_out,tend,nx,nz,hs,dt,i,k,ll)
end
@kernel function pack!(state,sl,sr,nx::Int32,nz::Int32,hs::Int32)
 positions=@index(Global,NTuple)
 s=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 ll=Int32(positions[3])-Int32(1)
 pack_body!(state,sl,sr,nx,nz,hs,s,k,ll)
end
@kernel function unpack!(state,rl,rr,nx::Int32,nz::Int32,hs::Int32)
 positions=@index(Global,NTuple)
 s=Int32(positions[1])-Int32(1)
 k=Int32(positions[2])-Int32(1)
 ll=Int32(positions[3])-Int32(1)
 unpack_body!(state,rl,rr,nx,nz,hs,s,k,ll)
end
@kernel function mass!(mass,te,state,h0,h1,nx::Int32,nz::Int32,dx::Float64,dz::Float64)
 positions=@index(Global,NTuple);i=Int32(positions[1])-Int32(1);k=Int32(positions[2])-Int32(1)
 if i<nx && k<nz
  m,e=mass_terms_body(state,h0,h1,nx,nz,dx,dz,i,k)
  Atomix.@atomic mass[1]+=m
  Atomix.@atomic te[1]+=e
 end
end
function launch(which,args,blocks,threads)
 if which==:zhalo
  zhalo!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:fx
  fx!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:tx
  tx!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:fz
  fz!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:tz
  tz!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:update
  update!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:pack
  pack!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:unpack
  unpack!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 elseif which==:mass
  mass!(BACKEND,threads)(args...;ndrange=threads .* blocks)
 else;error("Unknown sourcekernel");end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
