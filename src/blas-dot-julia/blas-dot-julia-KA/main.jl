using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function dot_kernel!(a,b,result,::Type{Acc}) where {Acc}
    i=@index(Local,Linear)
    scratch=@localmem Acc (256,)
    @inbounds scratch[i]=Acc(a[i])*Acc(b[i])
    @synchronize
    offset=128
    while offset>=1
        if i<=offset
            @inbounds scratch[i]+=scratch[i+offset]
        end
        @synchronize
        offset÷=2
    end
    if i==1
        @inbounds result[1]=eltype(result)(scratch[1])
    end
end
function study_prepare(a,b,result,n)
    T=eltype(a);Acc=T==Float64 ? Float64 : Float32;k=dot_kernel!(CUDABackend(),256)
    ()->k(a,b,result,Acc;ndrange=256)
end
exit(study_main(study_prepare))
