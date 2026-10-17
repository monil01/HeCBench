using CUDA,KernelAbstractions
@kernel function stream_map!(a,b,c,op)
    i=@index(Global,Linear)
    @inbounds if op==1
        c[i]=a[i]
    elseif op==2
        b[i]=0.4f0*c[i]
    elseif op==3
        c[i]=a[i]+b[i]
    elseif op==4
        a[i]=b[i]+0.4f0*c[i]
    elseif op==5
        a[i]+=b[i]+0.4f0*c[i]
    end
end
@kernel function stream_dot!(a,b,partial,n)
    lid=@index(Local,Linear);group=@index(Group,Linear);i=@index(Global,Linear)
    scratch=@localmem Float32 (256,)
    acc=0f0
    @inbounds while i<=n
        acc+=a[i]*b[i];i+=65536
    end
    scratch[lid]=acc
    offset=128
    while offset>0
        @synchronize
        if lid<=offset
            @inbounds scratch[lid]+=scratch[lid+offset]
        end
        offset÷=2
    end
    if lid==1
        @inbounds partial[group]=scratch[1]
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(op,a,b,c,partial,n)
    op=="dot" && return ()->stream_dot!(CUDABackend(),256)(a,b,partial,n;ndrange=65536)
    code=findfirst(==(op),["copy","mul","add","triad","nstream"]);isnothing(code)&&error("operation")
    ()->stream_map!(CUDABackend(),256)(a,b,c,code;ndrange=n)
end
stream_study(factory)
