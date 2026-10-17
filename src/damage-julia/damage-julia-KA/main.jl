using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function damage_kernel!(nlist,family,count,damage,n::Int32)
 lid=Int32(@index(Local,Linear));group=Int32(@index(Group,Linear));i=(group-Int32(1))*Int32(256)+lid
 scratch=@localmem Int32 (256,)
 @inbounds scratch[lid]=(i<=n&&nlist[i]!=Int32(-1)) ? Int32(1) : Int32(0)
 @synchronize
 stride=Int32(128)
 while stride>0
  if lid<=stride;@inbounds scratch[lid]+=scratch[lid+stride];end
  @synchronize;stride=div(stride,Int32(2))
 end
 if lid==Int32(1)
  @inbounds count[group]=scratch[1]
  @inbounds damage[group]=1.0-Float64(scratch[1])/Float64(family[group])
 end
end
function factory(input,family,count,damage,n,m)
 kernel=damage_kernel!(get_backend(input),256)
 return ()->kernel(input,family,count,damage,Int32(n);ndrange=m*256)
end
damage_study(factory)
