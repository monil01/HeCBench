using KernelAbstractions

include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function likelihood_kernel!(n,k,ax,ay,xj,yj,ind,obj,like,im,w,seeds,part)
 t=Int32(@index(Local,Linear))-Int32(1);b=Int32(@index(Group,Linear))-Int32(1);i=b*Int32(256)+t;shared=@localmem Float32 (256,)
 motion!(i,n,ax,ay,xj,yj,seeds);@synchronize;likelihood!(i,n,k,ax,ay,ind,obj,like,im,w);@inbounds shared[t+1]=i<n ? w[i+17] : 0f0;@synchronize
 for s in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1));if t<s;@inbounds shared[t+1]+=shared[t+s+1];end;@synchronize;end
 if t==0;@inbounds part[b+17]=shared[1];end;nothing
end
@kernel function sum_kernel!(n,part);sum_partials!(n,part);nothing;end
@kernel function scale_kernel!(n,w,part);i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear))-Int32(1);scale!(i,n,w,part);nothing;end
@kernel function cdf_kernel!(n,w,cdf,u,seeds)
 t=Int32(@index(Local,Linear))-Int32(1);shared=@localmem Float32 (1,)
 if t==0;cdf_offset!(n,w,cdf,u,seeds);end;@synchronize;if t==0;@inbounds shared[1]=u[17];end;@synchronize;thresholds!(t,n,u,shared[1]);nothing
end
@kernel function resample_kernel!(n,ax,ay,cdf,u,xj,yj);i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear))-Int32(1);resample!(i,n,ax,ay,cdf,u,xj,yj);nothing;end

const BACKEND=KernelAbstractions.get_backend(CUDA.zeros(UInt8,1))
function launch(a,k)
 n=a.n;b=cld(n,Int32(256));ax,ay,ll,w,cdf,u,xj,yj,part=a.dev
 likelihood_kernel!(BACKEND,256)(n,k,ax,ay,xj,yj,a.ind,a.obj,ll,a.im,w,a.seed,part;ndrange=Int(b)*256)
 sum_kernel!(BACKEND,1)(n,part;ndrange=1)
 scale_kernel!(BACKEND,256)(n,w,part;ndrange=Int(b)*256)
 cdf_kernel!(BACKEND,256)(n,w,cdf,u,a.seed;ndrange=256)
 resample_kernel!(BACKEND,256)(n,ax,ay,cdf,u,xj,yj;ndrange=Int(b)*256)
end
run_study(launch)
