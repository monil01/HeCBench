using CUDA,Libdl,SHA,KernelAbstractions
include(joinpath(@__DIR__,"..","mini_device.jl"))
const MINI_KA_BACKEND=CUDA.CUDABackend(always_inline=true)
@kernel function vector_entry!(x,y,w,n,a,b,op)
    gi=@index(Global,Linear)
    mini_vector!(Int32(gi),n,x,y,w,a,b,op)
end
@kernel function matvec_entry!(rp,c,a,x,w,n)
    gi=@index(Global,Linear)
    mini_matvec!(Int32(gi),n,rp,c,a,x,w)
end
@kernel function dot_entry!(x,y,d,n,stride)
    ti=@index(Local,Linear);gi=@index(Global,Linear);bi=@index(Group,Linear)
    t=Int32(ti);i=Int32(gi);s=0.0
    @inbounds while i<=n
        s=muladd(x[i],y[i],s);i+=stride
    end
    red=@localmem Float64 (256,)
    @inbounds red[t]=s
    offset=Int32(128)
    while offset>0
        @synchronize
        if t<=offset
            @inbounds s+=red[t+offset]
            @inbounds red[t]=s
        end
        offset=offset÷Int32(2)
    end
    if t==1;@inbounds d[bi]=s;end
end
@kernel function final_entry!(d)
    ti=@index(Local,Linear);t=Int32(ti);@inbounds s=d[t]
    red=@localmem Float64 (256,)
    @inbounds red[t]=s
    offset=Int32(128)
    while offset>0
        @synchronize
        if t<=offset
            @inbounds s+=red[t+offset]
            @inbounds red[t]=s
        end
        offset=offset÷Int32(2)
    end
    if t==1;@inbounds d[1]=s;end
end
function mini_vector_launch(x,y,w,n,a,b,g,op)
    vector_entry!(MINI_KA_BACKEND,256)(x,y,w,n,a,b,op;ndrange=g*256)
end
function mini_matvec_launch(rp,c,a,x,w,n,g)
    matvec_entry!(MINI_KA_BACKEND,256)(rp,c,a,x,w,n;ndrange=g*256)
end
function mini_dot_launch(x,y,w,n,g)
    dot_entry!(MINI_KA_BACKEND,256)(x,y,w,n,Int32(g*256);ndrange=g*256)
end
function mini_final_launch(w)
    final_entry!(MINI_KA_BACKEND,256)(w;ndrange=256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mini_study()
