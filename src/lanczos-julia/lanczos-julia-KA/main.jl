using CUDA,KernelAbstractions
const LC_BACKEND=CUDA.CUDABackend(always_inline=true)
@kernel function ka_dot!(n,x,y,z)
    ti=@index(Local,Linear);gi=@index(Global,Linear);bi=@index(Group,Linear)
    t=Int32(ti);i=Int32(gi);s=@localmem Float64 (256,)
    @inbounds s[t]=i<=n ? x[i]*y[i] : 0.0
    @synchronize
    h=Int32(128)
    while h>0
        if t<=h;@inbounds s[t]=s[t]+s[t+h];end
        @synchronize
        h=h÷Int32(2)
    end
    if t==1;@inbounds z[bi]=s[1];end
end
@kernel function ka_scale!(n,x,a)
    gi=@index(Global,Linear);i=Int32(gi)
    if i<=n;@inbounds x[i]=x[i]*a;end
end
@kernel function ka_axpy!(n,y,x,a)
    gi=@index(Global,Linear);i=Int32(gi)
    if i<=n;@inbounds y[i]=muladd(a,x[i],y[i]);end
end
@kernel function ka_spmv!(n,g,rp,ci,v,x,y)
    ti=@index(Local,Linear);gi=@index(Global,Linear);t=Int32(ti);i=Int32(gi)
    row=(i-Int32(1))÷g+Int32(1);lane=(i-Int32(1))%g
    s=@localmem Float64 (256,)
    @inbounds s[t]=0.0
    if row<=n
        @inbounds j=rp[row]+lane+Int32(1);@inbounds stop=rp[row+Int32(1)]
        while j<=stop
            @inbounds s[t]=muladd(v[j],x[ci[j]+Int32(1)],s[t]);j+=g
        end
    end
    @synchronize
    h=g÷Int32(2)
    while h>0
        if lane<h;@inbounds s[t]=s[t]+s[t+h];end
        @synchronize
        h=h÷Int32(2)
    end
    if row<=n&&lane==0;@inbounds y[row]=s[t];end
end
function launch_lanczos(stage,n,nz,g,blocks,scalar,a)
    extent=Int(blocks)*256
    if stage==0
        ka_dot!(LC_BACKEND,256)(n,a[1],a[2],a[3];ndrange=extent)
    elseif stage==1
        ka_scale!(LC_BACKEND,256)(n,a[1],scalar;ndrange=extent)
    elseif stage==2
        ka_axpy!(LC_BACKEND,256)(n,a[1],a[2],scalar;ndrange=extent)
    else
        ka_spmv!(LC_BACKEND,256)(n,g,a[1],a[2],a[3],a[4],a[5];ndrange=extent)
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
lanczos_study()
