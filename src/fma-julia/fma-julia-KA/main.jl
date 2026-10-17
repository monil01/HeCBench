using CUDA,KernelAbstractions
@kernel function fma_basic_ka!(a,b,c,ins,outs,ops,ch)
    i=@index(Global,Linear)-1
    if i<ops*ch
        op=i÷ch;k=i%ch
        @inbounds c[outs[op+1]*ch+k+1]+=a[ins[op+1]*ch+k+1]*b[k+1]
    end
end
@kernel function fma_row_ka!(a,b,c,ins,outs,ops,ch)
    op=@index(Global,Linear)
    if op<=ops
        for k in 0:ch-1
            @inbounds c[outs[op]*ch+k+1]+=a[ins[op]*ch+k+1]*b[k+1]
        end
    end
end
function launch_ka!(a,b,c1,c2,ins,outs,ops,ch,na,nc)
    fma_basic_ka!(get_backend(c1),256)(a,b,c1,ins,outs,ops,ch;ndrange=Int(ops*ch))
    fma_row_ka!(get_backend(c2),256)(a,b,c2,ins,outs,ops,ch;ndrange=Int(ops))
end
include(joinpath(@__DIR__,"..","study_common.jl"));fma_study(launch_ka!)
