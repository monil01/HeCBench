using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function fresnel_kernel!(idx,input,output,n)
    i=Int32(idx)
    if i<=n
        @inbounds output[i]=fresnel_corrected(input[i])
    end
    nothing
end
function study_prepare(input,output,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(fresnel_kernel!,spec,n,input,output,Int32(n))
end
exit(study_main(study_prepare))
