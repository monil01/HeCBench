using JACC
include(joinpath(@__DIR__,"..","gaussian_device.jl"))
@inline function seed_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1)
    means=CUDA.CuStaticSharedArray(Float32,24)
    variances=CUDA.CuStaticSharedArray(Float32,24)
    avg=CUDA.CuStaticSharedArray(Float32,1)
    if tid<d
        sum=0f0
        for e in Int32(0):n-Int32(1);sum+=gmm_at(data,e*d+tid);end
        @inbounds means[tid+Int32(1)]=sum/Float32(n)
    end
    CUDA.sync_threads()
    if tid<d
        sum=0f0
        for e in Int32(0):n-Int32(1);v=gmm_at(data,e*d+tid);sum=muladd(v,v,sum);end
        mu=@inbounds means[tid+Int32(1)]
        @inbounds variances[tid+Int32(1)]=muladd(-mu,mu,sum/Float32(n))
    end
    CUDA.sync_threads()
    if tid==0
        total=0f0
        for k in Int32(0):d-Int32(1);total+=(@inbounds variances[k+Int32(1)]);end
        @inbounds avg[1]=total/Float32(d)
    end
    CUDA.sync_threads()
    seed=m>1 ? (Float32(n)-1f0)/(Float32(m)-1f0) : 0f0
    for cluster in Int32(0):m-Int32(1)
        if tid<d;gmm_put!(c.means,cluster*d+tid,gmm_at(data,unsafe_trunc(Int32,Float32(cluster)*seed)*d+tid));end
        for i in tid:Int32(256):d*d-Int32(1);gmm_put!(c.R,cluster*d*d+i,div(i,d)==rem(i,d) ? 1f0 : 0f0);end
        if tid==0
            gmm_put!(c.pi,cluster,1f0/Float32(m));gmm_put!(c.N,cluster,Float32(n)/Float32(m))
            gmm_put!(c.avgvar,cluster,Float32(Float64(@inbounds avg[1])/1e6))
        end
    end
    return
end
@inline function constants_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1);cluster=Int32(blockIdx().x)-Int32(1)
    matrix=CUDA.CuStaticSharedArray(Float32,576);det=CUDA.CuStaticSharedArray(Float32,1);sum=CUDA.CuStaticSharedArray(Float32,1)
    for i in tid:Int32(256):d*d-Int32(1);gmm_put!(matrix,i,gmm_at(c.R,cluster*d*d+i));end
    CUDA.sync_threads()
    if tid==0;gmm_invert!(matrix,d,det);end
    CUDA.sync_threads()
    for i in tid:Int32(256):d*d-Int32(1);gmm_put!(c.Rinv,cluster*d*d+i,gmm_at(matrix,i));end
    CUDA.sync_threads()
    if tid==0
        gmm_put!(c.constant,cluster,muladd(-Float32(d)*.5f0,log(2f0*Float32(pi)),-.5f0*(@inbounds det[1])))
    end
    CUDA.sync_threads()
    if cluster==0
        if tid==0
            total=0f0
            for i in Int32(0):m-Int32(1);total+=gmm_at(c.N,i);end
            @inbounds sum[1]=total
        end
        CUDA.sync_threads()
        for i in tid:Int32(256):m-Int32(1)
            count=gmm_at(c.N,i);gmm_put!(c.pi,i,count<.5f0 ? 1f-10 : count/(@inbounds sum[1]))
        end
    end
    return
end
@inline function estep1_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1);cluster=Int32(blockIdx().x)-Int32(1);group=Int32(blockIdx().y)-Int32(1)
    means=CUDA.CuStaticSharedArray(Float32,24);rinv=CUDA.CuStaticSharedArray(Float32,576)
    if tid<d;gmm_put!(means,tid,gmm_at(c.means,cluster*d+tid));end
    for i in tid:Int32(256):d*d-Int32(1);gmm_put!(rinv,i,gmm_at(c.Rinv,cluster*d*d+i));end
    weight=gmm_at(c.pi,cluster);constant=gmm_at(c.constant,cluster)
    step=div(n,Int32(24));step-=rem(step,Int32(16));start=group*step+tid;stop=group==23 ? n : (group+Int32(1))*step
    CUDA.sync_threads()
    for e in start:Int32(256):stop-Int32(1)
        like=0f0
        for i in Int32(0):d-Int32(1),j in Int32(0):d-Int32(1)
            x=gmm_at(data,i*n+e)-gmm_at(means,i);y=gmm_at(data,j*n+e)-gmm_at(means,j)
            like=muladd(x*y,gmm_at(rinv,i*d+j),like)
        end
        gmm_put!(c.memberships,cluster*n+e,muladd(-.5f0,like,constant)+log(weight))
    end
    return
end
@inline function estep2_kernel(i,j,data,c,d,m,n,likelihood)
    tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
    totals=CUDA.CuStaticSharedArray(Float32,256)
    step=div(n,Int32(24));step-=rem(step,Int32(16));start=group*step+tid;stop=group==23 ? n : (group+Int32(1))*step
    sum=0f0
    for e in start:Int32(256):stop-Int32(1)
        maxlike=gmm_at(c.memberships,e)
        for cluster in Int32(1):m-Int32(1);maxlike=max(maxlike,gmm_at(c.memberships,cluster*n+e));end
        denom=0f0
        for cluster in Int32(0):m-Int32(1);denom+=exp(gmm_at(c.memberships,cluster*n+e)-maxlike);end
        denom=maxlike+log(denom);sum+=denom
        for cluster in Int32(0):m-Int32(1);gmm_put!(c.memberships,cluster*n+e,exp(gmm_at(c.memberships,cluster*n+e)-denom));end
    end
    @inbounds totals[tid+Int32(1)]=sum
    CUDA.sync_threads();sum=gmm_xorsum!(totals,tid,Int32(0))
    if tid==0;gmm_put!(likelihood,group,sum);end
    return
end
@inline function count_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1);cluster=Int32(blockIdx().x)-Int32(1)
    buf=CUDA.CuStaticSharedArray(Float32,256);sum=0f0
    for e in tid:Int32(256):n-Int32(1);sum+=gmm_at(c.memberships,cluster*n+e);end
    @inbounds buf[tid+Int32(1)]=sum
    CUDA.sync_threads();sum=gmm_xorsum!(buf,tid,Int32(0))
    if tid==0;gmm_put!(c.N,cluster,sum);gmm_put!(c.pi,cluster,sum);end
    return
end
@inline function means_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1);cluster=Int32(blockIdx().x)-Int32(1);dim=Int32(blockIdx().y)-Int32(1)
    buf=CUDA.CuStaticSharedArray(Float32,256);sum=0f0
    for e in tid:Int32(256):n-Int32(1);sum=muladd(gmm_at(data,dim*n+e),gmm_at(c.memberships,cluster*n+e),sum);end
    @inbounds buf[tid+Int32(1)]=sum
    CUDA.sync_threads();sum=gmm_xorsum!(buf,tid,Int32(0))
    if tid==0;gmm_put!(c.means,cluster*d+dim,sum);end
    return
end
@inline function covariance_kernel(i,j,data,c,d,m,n)
    tid=Int32(threadIdx().x)-Int32(1);first=(Int32(blockIdx().x)-Int32(1))*Int32(6)
    row,col=gmm_cell2(Int32(blockIdx().y)-Int32(1),d)
    meansrow=CUDA.CuStaticSharedArray(Float32,6);meanscol=CUDA.CuStaticSharedArray(Float32,6);buf=CUDA.CuStaticSharedArray(Float32,1536)
    CUDA.sync_threads()
    if tid<min(m,Int32(6))&&first+tid<m
        gmm_put!(meansrow,tid,gmm_at(c.means,(first+tid)*d+row));gmm_put!(meanscol,tid,gmm_at(c.means,(first+tid)*d+col))
    end
    CUDA.sync_threads()
    sums=MVector{6,Float32}(0f0,0f0,0f0,0f0,0f0,0f0)
    for q in Int32(0):Int32(5);gmm_put!(buf,q*Int32(256)+tid,0f0);end
    for e in tid:Int32(256):n-Int32(1)
        x=gmm_at(data,row*n+e);y=gmm_at(data,col*n+e)
        for q in Int32(0):Int32(5)
            if first+q<m
                v=(x-gmm_at(meansrow,q))*(y-gmm_at(meanscol,q))
                @inbounds sums[q+Int32(1)]=muladd(v,gmm_at(c.memberships,(first+q)*n+e),sums[q+Int32(1)])
            end
        end
    end
    for q in Int32(0):Int32(5);gmm_put!(buf,q*Int32(256)+tid,@inbounds sums[q+Int32(1)]);end
    CUDA.sync_threads()
    for q in Int32(0):Int32(5)
        value=gmm_xorsum!(buf,tid,q*Int32(256));gmm_put!(buf,q*Int32(256)+tid,value);CUDA.sync_threads()
    end
    if tid==0
        for q in Int32(0):min(Int32(5),m-first-Int32(1))
            offset=(first+q)*d*d;value=gmm_at(buf,q*Int32(256));gmm_put!(c.R,offset+row*d+col,value);gmm_put!(c.R,offset+col*d+row,value)
            if row==col;gmm_put!(c.R,offset+row*d+col,value+gmm_at(c.avgvar,first+q));end
        end
    end
    return
end
# Full explicit immutable-per-key shapes avoid inferred-grid mutation/reuse.
const GMM_SPECS=Dict{Tuple{Int,Int},Any}()
const GMM_SHAPES=Dict{Tuple{Int,Int},Tuple{Int,Int}}()
for stage in 1:7, m in 1:6
    gx=stage==1 ? 1 : stage==4 ? 24 : stage==7 ? cld(m,6) : m
    gy=stage==3 ? 24 : stage==6 ? 2 : stage==7 ? 3 : 1
    GMM_SPECS[(stage,m)]=JACC.launch_spec(;threads=(256,1),blocks=(gx,gy),shmem_size=0,sync=false)
    GMM_SHAPES[(stage,m)]=(gx*256,gy)
end
function launch_gmm(stage,gx,gy,threads,data,c,d,m,n,likelihood)
    key=(Int(stage),Int(m));spec=GMM_SPECS[key];shape=GMM_SHAPES[key]
    @assert shape==(Int(gx)*256,Int(gy))
    if stage==1;JACC.parallel_for(spec,shape,seed_kernel,data,c,d,m,n)
    elseif stage==2;JACC.parallel_for(spec,shape,constants_kernel,data,c,d,m,n)
    elseif stage==3;JACC.parallel_for(spec,shape,estep1_kernel,data,c,d,m,n)
    elseif stage==4;JACC.parallel_for(spec,shape,estep2_kernel,data,c,d,m,n,likelihood)
    elseif stage==5;JACC.parallel_for(spec,shape,count_kernel,data,c,d,m,n)
    elseif stage==6;JACC.parallel_for(spec,shape,means_kernel,data,c,d,m,n)
    elseif stage==7;JACC.parallel_for(spec,shape,covariance_kernel,data,c,d,m,n)
    else;error("unknownsourcekernel");end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
gmm_study()
