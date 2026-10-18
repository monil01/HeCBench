using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
const BACKEND=CUDA.CUDABackend()
@kernel function che_kernel!(stage,args...)
 xx,yy,zz=@index(Global,NTuple)
 point!(Int32(xx-1),Int32(yy-1),Int32(zz-1),stage,args...)
end
function launch(stage,cold,cnew,mu,ff,nx,ny,nz)
 nd=(cld(Int(nx),16)*16,cld(Int(ny),4)*4,cld(Int(nz),4)*4)
 che_kernel!(BACKEND,(16,4,4))(Val(stage),cold,cnew,mu,ff,nx,ny,nz;ndrange=nd)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
