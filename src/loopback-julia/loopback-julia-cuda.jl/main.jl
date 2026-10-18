using CUDA
include(joinpath(@__DIR__,"..","lookback_math.jl"))
function lookback!(args)
 path=CuStaticSharedArray(Float32,3840)
 lookback_site!(Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1),Int32(threadIdx().x),args...,path)
 return
end
function probe!(draws,seeds,out)
 rng_probe!(Int32(threadIdx().x-1),draws,seeds,out)
 return
end
function launch(c)
 @cuda threads=256 blocks=1 lookback!(c.args)
 nothing
end
function probe(c)
 @cuda threads=256 blocks=1 probe!(Int32(512*2*cld(Int(c.T),2)),c.seeds,c.probe)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
lookback_study(launch,probe)
