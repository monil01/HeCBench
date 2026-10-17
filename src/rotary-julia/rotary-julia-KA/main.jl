using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_rotary!(o1,o2,x1,x2,cosv,sinv,n::Int64)
    block_start=Int64(@index(Group,Linear)-1)*Int64(512);lane=Int64(@index(Local,Linear)-1)
    @inbounds for work in Int64(0):Int64(3)
        idx0=block_start+lane+work*Int64(128)
        if idx0<n
            idx=idx0+Int64(1);a=x1[idx];b=x2[idx];c=cosv[idx];s=sinv[idx];o1[idx]=a*c-b*s;o2[idx]=a*s+b*c
        end
    end
    nothing
end
function study_prepare(o1,o2,a,b,c,s,n)
    kernel=portable_rotary!(CUDABackend(),128);()->kernel(o1,o2,a,b,c,s,Int64(n);ndrange=cld(n,512)*128)
end
exit(study_main(study_prepare))
