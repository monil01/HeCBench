using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function launch_calculate_force_kernel!(::Val{N},phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.calculate_force_kernel!(z,y,x,phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0)
 return
end
function launch_allen_cahn_kernel!(::Val{N},phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.allen_cahn_kernel!(z,y,x,phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz)
 return
end
function launch_boundary_phi_kernel!(::Val{N},phinew) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.boundary_phi_kernel!(z,y,x,phinew)
 return
end
function launch_thermal_kernel!(::Val{N},unew,uold,phinew,phiold,d,dt,dx,dy,dz) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.thermal_kernel!(z,y,x,unew,uold,phinew,phiold,d,dt,dx,dy,dz)
 return
end
function launch_boundary_u_kernel!(::Val{N},unew,delta) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.boundary_u_kernel!(z,y,x,unew,delta)
 return
end
function launch_swap_kernel!(::Val{N},a,b) where {N}
 mod=N==16 ? Ace16 : Ace24
 z=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);x=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 mod.swap_kernel!(z,y,x,a,b)
 return
end
function dispatch(mod,n,stage,args...)
 b=(8,8,4);g=(n÷8,n÷8,n÷4)
 stage==1 && (@cuda threads=b blocks=g launch_calculate_force_kernel!(Val(n),args...);return)
 stage==2 && (@cuda threads=b blocks=g launch_allen_cahn_kernel!(Val(n),args...);return)
 stage==3 && (@cuda threads=b blocks=g launch_boundary_phi_kernel!(Val(n),args...);return)
 stage==4 && (@cuda threads=b blocks=g launch_thermal_kernel!(Val(n),args...);return)
 stage==5 && (@cuda threads=b blocks=g launch_boundary_u_kernel!(Val(n),args...);return)
 stage==6 && (@cuda threads=b blocks=g launch_swap_kernel!(Val(n),args...);return)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),dispatch)
