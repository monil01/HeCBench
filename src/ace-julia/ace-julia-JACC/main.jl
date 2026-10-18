using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function launch_calculate_force_kernel!(zz,yy,xx,::Val{N},phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.calculate_force_kernel!(z,y,x,phi,fx,fy,fz,dx,dy,dz,epsilon,w0,tau0)
 return
end
function launch_allen_cahn_kernel!(zz,yy,xx,::Val{N},phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.allen_cahn_kernel!(z,y,x,phinew,phiold,uold,fx,fy,fz,epsilon,w0,tau0,lambda,dt,dx,dy,dz)
 return
end
function launch_boundary_phi_kernel!(zz,yy,xx,::Val{N},phinew) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.boundary_phi_kernel!(z,y,x,phinew)
 return
end
function launch_thermal_kernel!(zz,yy,xx,::Val{N},unew,uold,phinew,phiold,d,dt,dx,dy,dz) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.thermal_kernel!(z,y,x,unew,uold,phinew,phiold,d,dt,dx,dy,dz)
 return
end
function launch_boundary_u_kernel!(zz,yy,xx,::Val{N},unew,delta) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.boundary_u_kernel!(z,y,x,unew,delta)
 return
end
function launch_swap_kernel!(zz,yy,xx,::Val{N},a,b) where {N}
 z=Int32(zz-1);y=Int32(yy-1);x=Int32(xx-1)
 mod=N==16 ? Ace16 : Ace24
 mod.swap_kernel!(z,y,x,a,b)
 return
end
function dispatch(mod,n,stage,args...)
 spec=JACC.launch_spec(threads=(8,8,4),blocks=(n÷8,n÷8,n÷4),shmem_size=0,sync=false);nd=(n,n,n)
 stage==1 && (JACC.parallel_for(spec,nd,launch_calculate_force_kernel!,Val(n),args...);return)
 stage==2 && (JACC.parallel_for(spec,nd,launch_allen_cahn_kernel!,Val(n),args...);return)
 stage==3 && (JACC.parallel_for(spec,nd,launch_boundary_phi_kernel!,Val(n),args...);return)
 stage==4 && (JACC.parallel_for(spec,nd,launch_thermal_kernel!,Val(n),args...);return)
 stage==5 && (JACC.parallel_for(spec,nd,launch_boundary_u_kernel!,Val(n),args...);return)
 stage==6 && (JACC.parallel_for(spec,nd,launch_swap_kernel!,Val(n),args...);return)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),dispatch)
