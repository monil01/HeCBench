using CUDA
include(joinpath(@__DIR__,"..","ert_math.jl"))
function ert_kernel!(A,n,trials,total)
 start=UInt32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 ert_site!(start,total,A,n,trials);return
end
function launch_ert(model,A,n,t,blocks,threads)
 @cuda threads=threads blocks=blocks ert_kernel!(A,UInt32(n),UInt32(t),UInt32(blocks*threads))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:cuda_jl)
