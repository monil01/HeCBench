using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function cooling_kernel!(i,temp,result,n,heat)
    if i<=n
        @inbounds result[i]=primordial_cool(.0899,temp[i],heat)
    end
    nothing
end
function study_prepare(temp,result,n,heat)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(cooling_kernel!,spec,n,temp,result,Int32(n),heat)
end
exit(study_main(study_prepare))
