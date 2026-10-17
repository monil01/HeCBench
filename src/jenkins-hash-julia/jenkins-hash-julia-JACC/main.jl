using CUDA
using JACC
include(joinpath(@__DIR__,"..","jenkins_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function jenkins_kernel!(index,lengths,initvals,keys,out,n::Int64)
    id=Int64(index)
    if id<=n
        idx=Int(id)
        @inbounds out[idx]=hashlittle_words(keys,(idx-1)*16+1,lengths[idx],initvals[idx])
    end
    nothing
end
function study_prepare(lens,seeds,keys,out,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(jenkins_kernel!,spec,n,lens,seeds,keys,out,Int64(n))
end
exit(study_main(study_prepare))
