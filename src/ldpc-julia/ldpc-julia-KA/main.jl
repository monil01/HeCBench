using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","ldpc_math.jl"))
@kernel unsafe_indices=true function cnp_first!(llr,dt,R,count,compact)
 t=@index(Local,NTuple);b=@index(Group,NTuple)
 x=Int32(t[1]-1);y=Int32(t[2]-1);r=Int32(b[1]-1);mc=Int32(b[2]-1)
 s=cnp_prepare!(Val(true),x,y,r,mc,llr,R,count,compact,nothing)
 cnp_finish!(Val(true),r,dt,R,compact,nothing,s)
end
@kernel unsafe_indices=true function cnp_normal!(llr,dt,R,count,compact)
 t=@index(Local,NTuple);b=@index(Group,NTuple)
 x=Int32(t[1]-1);y=Int32(t[2]-1);r=Int32(b[1]-1);mc=Int32(b[2]-1);cache=@localmem Float32 (1344,)
 s=cnp_prepare!(Val(false),x,y,r,mc,llr,R,count,compact,cache)
 @synchronize
 cnp_finish!(Val(false),r,dt,R,compact,cache,s)
end
@kernel unsafe_indices=true function vnp!(last,llr,dt,hd,count,compact)
 t=@index(Local,NTuple);b=@index(Group,NTuple)
 vnp_site!(last,Int32(t[1]-1),Int32(t[2]-1),Int32(b[1]-1),Int32(b[2]-1),llr,dt,hd,count,compact)
end
const FIRST=cnp_first!(CUDABackend(),(96,2));const NORMAL=cnp_normal!(CUDABackend(),(96,2));const VNP=vnp!(CUDABackend(),(96,2))
function decode(c)
 for it in 1:10
  kernel=it==1 ? FIRST : NORMAL
  kernel(c.llr,c.dt,c.R,c.count1,c.comp1;ndrange=(96*12,2*c.mcw))
  VNP(Val(it==10),c.llr,c.dt,c.hd,c.count2,c.comp2;ndrange=(96*24,2*c.mcw))
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
ldpc_study(decode)
