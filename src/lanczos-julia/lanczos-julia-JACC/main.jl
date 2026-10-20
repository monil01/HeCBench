using CUDA,JACC
@inline function jacc_dot!(idx,n,x,y,z)
    t=Int32(threadIdx().x);i=Int32(idx);s=CuStaticSharedArray(Float64,256)
    @inbounds s[t]=i<=n ? x[i]*y[i] : 0.0
    sync_threads();h=Int32(128)
    while h>0
        if t<=h;@inbounds s[t]=s[t]+s[t+h];end
        sync_threads();h=h÷Int32(2)
    end
    if t==1;@inbounds z[blockIdx().x]=s[1];end
    nothing
end
@inline function jacc_scale!(idx,n,x,a)
    i=Int32(idx);if i<=n;@inbounds x[i]=x[i]*a;end
    nothing
end
@inline function jacc_axpy!(idx,n,y,x,a)
    i=Int32(idx);if i<=n;@inbounds y[i]=muladd(a,x[i],y[i]);end
    nothing
end
@inline function jacc_spmv!(idx,n,g,rp,ci,v,x,y)
    t=Int32(threadIdx().x);i=Int32(idx);row=(i-Int32(1))÷g+Int32(1);lane=(i-Int32(1))%g
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
    nothing
end
# Complete fixed grids; no inferred/mutated dimension reuse and no unwanted dynamic48KiB.
const LC_SPECS=Dict(b=>JACC.launch_spec(;threads=(256,),blocks=(b,),shmem_size=0,sync=false) for b in (2,3,5,9,17,33))
function launch_lanczos(stage,n,nz,g,blocks,scalar,a)
    spec=LC_SPECS[Int(blocks)];extent=Int(blocks)*256
    if stage==0
        JACC.parallel_for(spec,extent,jacc_dot!,n,a[1],a[2],a[3])
    elseif stage==1
        JACC.parallel_for(spec,extent,jacc_scale!,n,a[1],scalar)
    elseif stage==2
        JACC.parallel_for(spec,extent,jacc_axpy!,n,a[1],a[2],scalar)
    else
        JACC.parallel_for(spec,extent,jacc_spmv!,n,g,a[1],a[2],a[3],a[4],a[5])
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
lanczos_study()
