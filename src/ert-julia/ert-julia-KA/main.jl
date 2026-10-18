using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","ert_math.jl"))
@kernel unsafe_indices=true function ert_kernel!(A,n,trials,total)
 start=UInt32(@index(Global,Linear)-1)
 ert_site!(start,total,A,n,trials)
end
function launch_ert(model,A,n,t,blocks,threads)
 ert_kernel!(CUDABackend(),threads)(A,UInt32(n),UInt32(t),UInt32(blocks*threads);ndrange=blocks*threads)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:ka)
