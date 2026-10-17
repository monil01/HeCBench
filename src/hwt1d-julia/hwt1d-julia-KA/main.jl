using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_scale!(data,scratch,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds data[idx]=data[idx]/sqrt(Float32(n))
    end
    nothing
end
@kernel function portable_level!(data,scratch,n::Int32)
    idx=Int32(@index(Global,Linear))
    half=n÷Int32(2)
    if idx<=half
        @inbounds begin
            a=data[Int32(2)*idx-Int32(1)];b=data[Int32(2)*idx]
            scratch[idx]=(a+b)/sqrt(2f0);scratch[half+idx]=(a-b)/sqrt(2f0)
        end
    end
    nothing
end
function study_prepare(data,scratch,n)
    sk=portable_scale!(CUDABackend(),256);lk=portable_level!(CUDABackend(),256)
    scale! = ()->sk(data,scratch,Int32(n);ndrange=n)
    level! = len->lk(data,scratch,len;ndrange=cld(Int(len)÷2,256)*256)
    scale!,level!
end
exit(study_main(study_prepare))
