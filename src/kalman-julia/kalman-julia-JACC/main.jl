using CUDA,JACC
include(joinpath(@__DIR__,"..","kalman_math.jl"))
function kalman_full!(i,args)
 kalman_site!(Int32(i-1),args...)
 nothing
end
const SPECS=Dict{Int,Any}()
function launch(c,args)
 nb=cld(Int(c.ns),256)
 spec=get!(SPECS,nb) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=nb,shmem_size=0,sync=false)
 end
 JACC.parallel_for(kalman_full!,spec,nb*256,args)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
kalman_study(launch)
