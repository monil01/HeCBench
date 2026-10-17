using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function damage_kernel!(nlist,family,count,damage,n::Int32)
 lid=Int32(threadIdx().x);group=Int32(blockIdx().x);i=(group-Int32(1))*Int32(256)+lid
 scratch=CuStaticSharedArray(Int32,256)
 @inbounds scratch[lid]=(i<=n&&nlist[i]!=Int32(-1)) ? Int32(1) : Int32(0)
 sync_threads()
 stride=Int32(128)
 while stride>0
  if lid<=stride;@inbounds scratch[lid]+=scratch[lid+stride];end
  sync_threads();stride=div(stride,Int32(2))
 end
 if lid==Int32(1)
  @inbounds count[group]=scratch[1]
  @inbounds damage[group]=1.0-Float64(scratch[1])/Float64(family[group])
 end
 return
end
factory(input,family,count,damage,n,m)=()->(@cuda threads=256 blocks=m damage_kernel!(input,family,count,damage,Int32(n)))
if abspath(PROGRAM_FILE)==@__FILE__
 damage_study(factory;n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 262144,repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10)
end
