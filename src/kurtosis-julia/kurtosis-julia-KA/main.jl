using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function kreduce_ka!(input,out,n::Int32,mode)
 tid=Int32(@index(Local,Linear));block=Int32(@index(Group,Linear));i=Int32(@index(Global,Linear));tile=@localmem KResult (256,)
 @inbounds tile[tid]=i<=n ? kload(input,i,mode) : zero(KResult)
 @synchronize
 stride=Int32(128)
 while stride>0
  if tid<=stride;@inbounds tile[tid]=kmerge(tile[tid],tile[tid+stride]);end
  @synchronize
  stride>>=1
 end
 if tid==1;@inbounds out[block]=tile[1];end
end
function launch!(x,partial,out,n,B)
 k=kreduce_ka!(CUDABackend(),256)
 k(x,partial,Int32(n),Val(0);ndrange=B*256)
 k(partial,out,Int32(B),Val(1);ndrange=256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
kurtosis_study(launch!)
