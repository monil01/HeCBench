
include(joinpath(@__DIR__,"..","study_common.jl"))
function likelihood_kernel!(n,k,ax,ay,xj,yj,ind,obj,like,im,w,seeds,part)
 t=Int32(threadIdx().x)-Int32(1);b=Int32(blockIdx().x)-Int32(1);i=b*Int32(256)+t;shared=CuStaticSharedArray(Float32,256)
 motion!(i,n,ax,ay,xj,yj,seeds);sync_threads();likelihood!(i,n,k,ax,ay,ind,obj,like,im,w);@inbounds shared[t+1]=i<n ? w[i+17] : 0f0;sync_threads()
 for s in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1));if t<s;@inbounds shared[t+1]+=shared[t+s+1];end;sync_threads();end
 if t==0;@inbounds part[b+17]=shared[1];end;return
end
function sum_kernel!(n,part);sum_partials!(n,part);return;end
function scale_kernel!(n,w,part);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1);scale!(i,n,w,part);return;end
function cdf_kernel!(n,w,cdf,u,seeds)
 t=Int32(threadIdx().x)-Int32(1);shared=CuStaticSharedArray(Float32,1)
 if t==0;cdf_offset!(n,w,cdf,u,seeds);end;sync_threads();if t==0;@inbounds shared[1]=u[17];end;sync_threads();thresholds!(t,n,u,shared[1]);return
end
function resample_kernel!(n,ax,ay,cdf,u,xj,yj);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1);resample!(i,n,ax,ay,cdf,u,xj,yj);return;end
function launch(a,k)
 n=a.n;b=cld(n,Int32(256));ax,ay,ll,w,cdf,u,xj,yj,part=a.dev
 @cuda threads=256 blocks=b likelihood_kernel!(n,k,ax,ay,xj,yj,a.ind,a.obj,ll,a.im,w,a.seed,part)
 @cuda threads=1 blocks=1 sum_kernel!(n,part)
 @cuda threads=256 blocks=b scale_kernel!(n,w,part)
 @cuda threads=256 blocks=1 cdf_kernel!(n,w,cdf,u,a.seed)
 @cuda threads=256 blocks=b resample_kernel!(n,ax,ay,cdf,u,xj,yj)
end
run_study(launch)
