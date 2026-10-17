using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function damage_kernel!(index,template,nlist,family,count,damage,n::Int32)
 lid=Int32(mod(index-1,256)+1);group=Int32(div(index-1,256)+1);i=(group-Int32(1))*Int32(256)+lid
 scratch=JACC.shared(template)
 @inbounds scratch[lid]=(i<=n&&nlist[i]!=Int32(-1)) ? Int32(1) : Int32(0)
 JACC.sync_workgroup()
 stride=Int32(128)
 while stride>0
  if lid<=stride;@inbounds scratch[lid]+=scratch[lid+stride];end
  JACC.sync_workgroup();stride=div(stride,Int32(2))
 end
 if lid==Int32(1)
  @inbounds count[group]=scratch[1]
  @inbounds damage[group]=1.0-Float64(scratch[1])/Float64(family[group])
 end
 return
end
function factory(input,family,count,damage,n,m)
 template=CUDA.zeros(Int32,256);spec=JACC.launch_spec(;threads=256,shmem_size=1024,sync=false)
 return ()->JACC.parallel_for(spec,m*256,damage_kernel!,template,input,family,count,damage,Int32(n))
end
damage_study(factory)
