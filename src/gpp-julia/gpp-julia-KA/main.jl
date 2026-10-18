using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","reduction_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel unsafe_indices=true function reduce_gpp!(gx,gy,args)
 tx=@index(Local,NTuple);b=@index(Group,NTuple)
 gpp_thread!(Int32(tx[1]-1),Int32(b[1]-1),Int32(b[2]-1),gx,gy,args...)
end
function launch(c)
 reduce_gpp!(CUDABackend(),(32,1))(c.gx,c.gy,c.args;ndrange=(32Int(c.gx),Int(c.gy)))
end
gpp_study(launch)
