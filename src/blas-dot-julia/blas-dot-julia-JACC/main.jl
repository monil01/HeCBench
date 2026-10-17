using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function dot_kernel!(i,a,b,result,template)
    scratch=JACC.shared(template);Acc=eltype(template)
    @inbounds scratch[i]=Acc(a[i])*Acc(b[i])
    JACC.sync_workgroup()
    offset=128
    while offset>=1
        if i<=offset
            @inbounds scratch[i]+=scratch[i+offset]
        end
        JACC.sync_workgroup()
        offset÷=2
    end
    if i==1
        @inbounds result[1]=eltype(result)(scratch[1])
    end
    nothing
end
function study_prepare(a,b,result,n)
    T=eltype(a);Acc=T==Float64 ? Float64 : Float32;template=CUDA.zeros(Acc,256);backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=1,shmem_size=256*sizeof(Acc),sync=false)
    ()->JACC.parallel_for(dot_kernel!,spec,256,a,b,result,template)
end
exit(study_main(study_prepare))
