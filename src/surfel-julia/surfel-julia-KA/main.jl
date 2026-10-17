using CUDA, KernelAbstractions
@inline block_barrier()=KernelAbstractions.@synchronize()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function base!(s,N,f,w,h,d)
 xy=@index(Global,NTuple);base_body!(Int32(xy[1])-Int32(1),Int32(xy[2])-Int32(1),s,N,f,w,h,d)
end
@kernel cpu=false function tile!(s,N,f,w,h,d)
 xy=@index(Global,NTuple);ij=@index(Local,NTuple);shared=@localmem(Float32,(1792,));tile_body!(Int32(xy[1])-Int32(1),Int32(xy[2])-Int32(1),(Int32(ij[2])-Int32(1))*Int32(16)+Int32(ij[1])-Int32(1),s,N,f,w,h,d,shared)
end
function run!(tiled,s,N,f,w,h,d,tmp)
 fun=tiled ? tile! : base!
 fun(get_backend(s),(16,16))(s,N,f,w,h,d;ndrange=(32,32))
 return
end
study(run!)
