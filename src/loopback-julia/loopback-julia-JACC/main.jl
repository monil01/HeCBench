using CUDA,JACC
include(joinpath(@__DIR__,"..","lookback_math.jl"))
const PATH_TEMPLATE=CUDA.zeros(Float32,3840)
const SIM=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=1,shmem_size=15360,sync=false)
const RNG=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=1,shmem_size=0,sync=false)
function lookback!(i,args,template)
 path=JACC.shared(template)
 lookback_site!(Int32(i-1),Int32(i),args...,path)
 nothing
end
function probe!(i,draws,seeds,out)
 rng_probe!(Int32(i-1),draws,seeds,out)
 nothing
end
launch(c)=(JACC.parallel_for(lookback!,SIM,256,c.args,PATH_TEMPLATE);nothing)
probe(c)=(JACC.parallel_for(probe!,RNG,256,Int32(512*2*cld(Int(c.T),2)),c.seeds,c.probe);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
lookback_study(launch,probe)
