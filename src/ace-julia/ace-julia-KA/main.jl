using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
const BACKEND=CUDA.CUDABackend()
@kernel function launch_calculate_force_kernel!(::Val{N},phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.calculate_force_kernel!(z,y,x,phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0)
end
@kernel function launch_allen_cahn_kernel!(::Val{N},phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.allen_cahn_kernel!(z,y,x,phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz)
end
@kernel function launch_boundary_phi_kernel!(::Val{N},phinew) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.boundary_phi_kernel!(z,y,x,phinew)
end
@kernel function launch_thermal_kernel!(::Val{N},unew,uold,phinew,phiold,d,dt,dx,dy,dz) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.thermal_kernel!(z,y,x,unew,uold,phinew,phiold,d,dt,dx,dy,dz)
end
@kernel function launch_boundary_u_kernel!(::Val{N},unew,delta) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.boundary_u_kernel!(z,y,x,unew,delta)
end
@kernel function launch_swap_kernel!(::Val{N},a,b) where {N}
 zz,yy,xx=@index(Global,NTuple)
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.swap_kernel!(z,y,x,a,b)
end
function dispatch(mod,n,stage,args...)
 b=(8,8,4);nd=(n,n,n)
 stage==1 && (launch_calculate_force_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
 stage==2 && (launch_allen_cahn_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
 stage==3 && (launch_boundary_phi_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
 stage==4 && (launch_thermal_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
 stage==5 && (launch_boundary_u_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
 stage==6 && (launch_swap_kernel!(BACKEND,b)(Val(n),args...;ndrange=nd);return)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),dispatch)
