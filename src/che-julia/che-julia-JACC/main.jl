using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function che_kernel!(xx,yy,zz,stage,args...)
 point!(Int32(xx-1),Int32(yy-1),Int32(zz-1),stage,args...)
 return
end
function launch(stage,cold,cnew,mu,ff,nx,ny,nz)
 grid=(cld(Int(nx),16),cld(Int(ny),4),cld(Int(nz),4));nd=(grid[1]*16,grid[2]*4,grid[3]*4)
 spec=JACC.launch_spec(threads=(16,4,4),blocks=grid,shmem_size=0,sync=false)
 JACC.parallel_for(spec,nd,che_kernel!,Val(stage),cold,cnew,mu,ff,nx,ny,nz)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
