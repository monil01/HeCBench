using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_scale!(i,data,scratch,n::Int32)
    idx=Int32(i)
    if idx<=n
        @inbounds data[idx]=data[idx]/sqrt(Float32(n))
    end
    nothing
end
function portable_level!(i,data,scratch,n::Int32)
    idx=Int32(i)
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
    backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};ss=ST(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false);specs=[ST(;threads=256,blocks=cld(len÷2,256),shmem_size=0,sync=false) for len in [n>>j for j in 0:13]]
    scale! = ()->JACC.parallel_for(portable_scale!,ss,n,data,scratch,Int32(n))
    level! = len->JACC.parallel_for(portable_level!,specs[trailing_zeros(n)-trailing_zeros(Int(len))+1],cld(Int(len)÷2,256)*256,data,scratch,len)
    scale!,level!
end
exit(study_main(study_prepare))
