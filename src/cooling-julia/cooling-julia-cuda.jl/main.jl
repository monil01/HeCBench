using CUDA
include(joinpath(@__DIR__,"..","cooling_math.jl"))
function cooling_kernel!(temp,result,n,heat)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=n
        @inbounds result[i]=primordial_cool(.0899,temp[i],heat)
    end
    nothing
end
function study_prepare(temp,result,n,heat)
    ()->(@cuda threads=256 blocks=cld(n,256) cooling_kernel!(temp,result,Int32(n),heat))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
