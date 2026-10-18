using JACC

include(joinpath(@__DIR__,"..","study_common.jl"))
function likelihood_kernel!(g,n,k,ax,ay,xj,yj,ind,obj,like,im,w,seeds,part,template)
 t=Int32((g-1)%256);b=Int32((g-1)÷256);i=b*Int32(256)+t;shared=JACC.shared(template)
 motion!(i,n,ax,ay,xj,yj,seeds);JACC.sync_workgroup();likelihood!(i,n,k,ax,ay,ind,obj,like,im,w);@inbounds shared[t+1]=i<n ? w[i+17] : 0f0;JACC.sync_workgroup()
 for s in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1));if t<s;@inbounds shared[t+1]+=shared[t+s+1];end;JACC.sync_workgroup();end
 if t==0;@inbounds part[b+17]=shared[1];end;return
end
function sum_kernel!(g,n,part);sum_partials!(n,part);return;end
function scale_kernel!(g,n,w,part);i=(Int32((g-1)÷256))*Int32(256)+Int32((g-1)%256);scale!(i,n,w,part);return;end
function cdf_kernel!(g,n,w,cdf,u,seeds,template)
 t=Int32((g-1)%256);shared=JACC.shared(template)
 if t==0;cdf_offset!(n,w,cdf,u,seeds);end;JACC.sync_workgroup();if t==0;@inbounds shared[1]=u[17];end;JACC.sync_workgroup();thresholds!(t,n,u,shared[1]);return
end
function resample_kernel!(g,n,ax,ay,cdf,u,xj,yj);i=(Int32((g-1)÷256))*Int32(256)+Int32((g-1)%256);resample!(i,n,ax,ay,cdf,u,xj,yj);return;end

const RED_TEMPLATE=CUDA.zeros(Float32,256)
const U_TEMPLATE=CUDA.zeros(Float32,1)
function specification(t,b,s)
 JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=t,blocks=b,shmem_size=s,sync=false)
end
function launch(a,k)
 n=a.n;b=Int(cld(n,Int32(256)));ax,ay,ll,w,cdf,u,xj,yj,part=a.dev
 JACC.parallel_for(likelihood_kernel!,specification(256,b,1024),b*256,n,k,ax,ay,xj,yj,a.ind,a.obj,ll,a.im,w,a.seed,part,RED_TEMPLATE)
 JACC.parallel_for(sum_kernel!,specification(1,1,0),1,n,part)
 JACC.parallel_for(scale_kernel!,specification(256,b,0),b*256,n,w,part)
 JACC.parallel_for(cdf_kernel!,specification(256,1,4),256,n,w,cdf,u,a.seed,U_TEMPLATE)
 JACC.parallel_for(resample_kernel!,specification(256,b,0),b*256,n,ax,ay,cdf,u,xj,yj)
end
run_study(launch)
