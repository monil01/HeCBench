using CUDA
include(joinpath(@__DIR__,"..","fresnel_math.jl"))
function fresnel_kernel!(input,output,n)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
    if i<=n
        @inbounds output[i]=fresnel_corrected(input[i])
    end
    nothing
end
function study_prepare(input,output,n)
    ()->(@cuda threads=256 blocks=cld(n,256) fresnel_kernel!(input,output,Int32(n)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
