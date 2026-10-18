using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function zhalo!(state,nx::Int32,nz::Int32,hs::Int32)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 ll=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 update_state_z_body!(state,nx,nz,hs,i,ll)
 return
end
function fx!(state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 compute_flux_x_body!(state,flux,hy_dens_cell,hy_dens_theta_cell,hv_coef,nx,nz,hs,i,k)
 return
end
function tx!(flux,tend,nx::Int32,nz::Int32,dx::Int32)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 ll=(Int32(CUDA.blockIdx().z)-1)*Int32(CUDA.blockDim().z)+Int32(CUDA.threadIdx().z)-1
 compute_tend_x_body!(flux,tend,nx,nz,dx,i,k,ll)
 return
end
function fz!(state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef::Float64,nx::Int32,nz::Int32,hs::Int32)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 compute_flux_z_body!(state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,hv_coef,nx,nz,hs,i,k)
 return
end
function tz!(state,flux,tend,nx::Int32,nz::Int32,hs::Int32,dz::Int32)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 ll=(Int32(CUDA.blockIdx().z)-1)*Int32(CUDA.blockDim().z)+Int32(CUDA.threadIdx().z)-1
 compute_tend_z_body!(state,flux,tend,nx,nz,hs,dz,i,k,ll)
 return
end
function update!(state_init,state_out,tend,nx::Int32,nz::Int32,hs::Int32,dt::Float64)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 ll=(Int32(CUDA.blockIdx().z)-1)*Int32(CUDA.blockDim().z)+Int32(CUDA.threadIdx().z)-1
 update_fluid_state_body!(state_init,state_out,tend,nx,nz,hs,dt,i,k,ll)
 return
end
function pack!(state,sl,sr,nx::Int32,nz::Int32,hs::Int32)
 s=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 ll=(Int32(CUDA.blockIdx().z)-1)*Int32(CUDA.blockDim().z)+Int32(CUDA.threadIdx().z)-1
 pack_body!(state,sl,sr,nx,nz,hs,s,k,ll)
 return
end
function unpack!(state,rl,rr,nx::Int32,nz::Int32,hs::Int32)
 s=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 ll=(Int32(CUDA.blockIdx().z)-1)*Int32(CUDA.blockDim().z)+Int32(CUDA.threadIdx().z)-1
 unpack_body!(state,rl,rr,nx,nz,hs,s,k,ll)
 return
end
function mass!(mass,te,state,h0,h1,nx::Int32,nz::Int32,dx::Float64,dz::Float64)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 if i<nx && k<nz
  m,e=mass_terms_body(state,h0,h1,nx,nz,dx,dz,i,k)
  CUDA.@atomic mass[1]+=m
  CUDA.@atomic te[1]+=e
 end
 return
end
function launch(which,args,blocks,threads)
 if which==:zhalo
  CUDA.@cuda threads=threads blocks=blocks zhalo!(args...)
 elseif which==:fx
  CUDA.@cuda threads=threads blocks=blocks fx!(args...)
 elseif which==:tx
  CUDA.@cuda threads=threads blocks=blocks tx!(args...)
 elseif which==:fz
  CUDA.@cuda threads=threads blocks=blocks fz!(args...)
 elseif which==:tz
  CUDA.@cuda threads=threads blocks=blocks tz!(args...)
 elseif which==:update
  CUDA.@cuda threads=threads blocks=blocks update!(args...)
 elseif which==:pack
  CUDA.@cuda threads=threads blocks=blocks pack!(args...)
 elseif which==:unpack
  CUDA.@cuda threads=threads blocks=blocks unpack!(args...)
 elseif which==:mass
  CUDA.@cuda threads=threads blocks=blocks mass!(args...)
 else;error("Unknown sourcekernel");end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
