using CUDA
function lanczos_dot!(n,x,y,z)
    t=Int32(threadIdx().x);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+t
    s=CuStaticSharedArray(Float64,256)
    @inbounds s[t]=i<=n ? x[i]*y[i] : 0.0
    sync_threads();h=Int32(128)
    while h>0
        if t<=h;@inbounds s[t]=s[t]+s[t+h];end
        sync_threads();h=h÷Int32(2)
    end
    if t==1;@inbounds z[blockIdx().x]=s[1];end
    return
end
function lanczos_scale!(n,x,a)
    i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)
    if i<=n;@inbounds x[i]=x[i]*a;end
    return
end
function lanczos_axpy!(n,y,x,a)
    i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)
    if i<=n;@inbounds y[i]=muladd(a,x[i],y[i]);end
    return
end
function lanczos_spmv!(n,g,rp,ci,v,x,y)
    t=Int32(threadIdx().x);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+t
    row=(i-Int32(1))÷g+Int32(1);lane=(i-Int32(1))%g
    s=CuStaticSharedArray(Float64,256);@inbounds s[t]=0.0
    if row<=n
        @inbounds j=rp[row]+lane+Int32(1);@inbounds stop=rp[row+Int32(1)]
        while j<=stop
            @inbounds s[t]=muladd(v[j],x[ci[j]+Int32(1)],s[t]);j+=g
        end
    end
    sync_threads();h=g÷Int32(2)
    while h>0
        if lane<h;@inbounds s[t]=s[t]+s[t+h];end
        sync_threads();h=h÷Int32(2)
    end
    if row<=n&&lane==0;@inbounds y[row]=s[t];end
    return
end
function launch_lanczos(stage,n,nz,g,blocks,scalar,a)
    if stage==0
        CUDA.@cuda threads=256 blocks=blocks always_inline=true lanczos_dot!(n,a[1],a[2],a[3])
    elseif stage==1
        CUDA.@cuda threads=256 blocks=blocks always_inline=true lanczos_scale!(n,a[1],scalar)
    elseif stage==2
        CUDA.@cuda threads=256 blocks=blocks always_inline=true lanczos_axpy!(n,a[1],a[2],scalar)
    else
        CUDA.@cuda threads=256 blocks=blocks always_inline=true lanczos_spmv!(n,g,a[1],a[2],a[3],a[4],a[5])
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
lanczos_study()
