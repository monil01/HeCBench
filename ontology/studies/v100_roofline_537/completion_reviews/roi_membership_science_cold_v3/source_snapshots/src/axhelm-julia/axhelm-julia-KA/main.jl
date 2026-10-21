ENV["JULIA_CUDA_MEMORY_POOL"]="none"
using CUDA, KernelAbstractions
@inline block_barrier()=KernelAbstractions.@synchronize()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function scalar!(offset,G,D,L,Q,A)
 ij=@index(Local,NTuple);group=@index(Group,NTuple);s=@localmem(Float32,(256,));helm_body!(Int32(ij[1])-Int32(1),Int32(ij[2])-Int32(1),Int32(group[1])-Int32(1),offset,G,D,L,Q,A,s,Val(1))
end
@kernel cpu=false function fused3!(offset,G,D,L,Q,A)
 ij=@index(Local,NTuple);group=@index(Group,NTuple);s=@localmem(Float32,(640,));helm_body!(Int32(ij[1])-Int32(1),Int32(ij[2])-Int32(1),Int32(group[1])-Int32(1),offset,G,D,L,Q,A,s,Val(3))
end
function run!(fields,offset,G,D,L,Q,A)
 f=fields==1 ? scalar! : fused3!
 f(get_backend(Q),(8,8))(offset,G,D,L,Q,A;ndrange=(8STUDY_ELEMENTS,8))
 return
end
study(run!)
