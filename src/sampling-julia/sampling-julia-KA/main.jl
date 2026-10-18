
using KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function swap_value!(x,i)
 Atomix.@atomicswap :monotonic x[i]=1f0
end
@kernel function exact_kernel!(cols,nbg,x,bg,obs,data,step)
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1);exact_site!(tid,row,step,cols,nbg,x,bg,obs,data)
end
@kernel function sampled_kernel!(exact,cols,nbg,x,bg,obs,data,ns,seed,step)
 tid=Int32(@index(Local,Linear))-Int32(1);blk=Int32(@index(Group,Linear))-Int32(1)
 claim_site!(tid,blk,exact,cols,x,ns,seed);@synchronize;sample_scatter!(tid,blk,step,exact,cols,nbg,x,bg,obs,data)
end
const BACKEND=KernelAbstractions.get_backend(CUDA.zeros(UInt8,1))
function launch(a)
 threads=Int(min(Int32(256),a.cols));step=Int32(threads)
 if a.exact>0;exact_kernel!(BACKEND,threads)(a.cols,a.nbg,a.x,a.bg,a.obs,a.data,step;ndrange=Int(a.exact)*threads);end
 if a.sampled>0;sampled_kernel!(BACKEND,threads)(a.exact,a.cols,a.nbg,a.x,a.bg,a.obs,a.data,a.ns,a.seed,step;ndrange=Int(a.sampled÷Int32(2))*threads);end
end
run_study(launch)
