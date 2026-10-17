using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_copy!(i,dst,src,n::Int32)
    idx=Int32(i)
    if idx<=n
        @inbounds dst[idx]=src[idx]
    end
    nothing
end
function study_prepare(dst,src,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_copy!,spec,cld(n,256)*256,dst,src,Int32(n))
end
exit(study_main(study_prepare))
