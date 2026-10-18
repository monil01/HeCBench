using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","lookback_math.jl"))
@kernel unsafe_indices=true function lookback!(args)
 path=@localmem Float32 (3840,)
 lookback_site!(Int32(@index(Global,Linear)-1),Int32(@index(Local,Linear)),args...,path)
end
@kernel unsafe_indices=true function probe!(draws,seeds,out)
 rng_probe!(Int32(@index(Global,Linear)-1),draws,seeds,out)
end
const SIM=lookback!(CUDABackend(),256);const RNG=probe!(CUDABackend(),256)
launch(c)=(SIM(c.args;ndrange=256);nothing)
probe(c)=(RNG(Int32(512*2*cld(Int(c.T),2)),c.seeds,c.probe;ndrange=256);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
lookback_study(launch,probe)
