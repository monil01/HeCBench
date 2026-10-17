using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function sobol_ka!(dirs,out,n,d,gx)
 tx=Int32(@index(Local,Linear))-Int32(1)
 group=@index(Group,NTuple);bx=Int32(group[1])-Int32(1);by=Int32(group[2])-Int32(1)
 tile=@localmem UInt32 (32,)
 if tx<Int32(32)
  @inbounds tile[tx+Int32(1)]=dirs[by*Int32(32)+tx+Int32(1)]
 end
 @synchronize
 sobol_body!(tx,bx,by,gx,tile,out,n)
end
const BACKEND=CUDABackend()
function run!(dirs,out,n,d,gx,template)
 sobol_ka!(BACKEND,(128,1))(dirs,out,n,d,gx;ndrange=(Int(gx)*128,Int(d)))
 nothing
end
study(run!)
