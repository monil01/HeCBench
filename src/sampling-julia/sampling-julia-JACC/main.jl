
using JACC,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function swap_value!(x,i)
 Atomix.@atomicswap :monotonic x[i]=1f0
end
function exact_kernel!(i,cols,nbg,x,bg,obs,data,step)
 tid=Int32((i-1)%step);row=Int32((i-1)÷step);exact_site!(tid,row,step,cols,nbg,x,bg,obs,data);nothing
end
function sampled_kernel!(i,exact,cols,nbg,x,bg,obs,data,ns,seed,step)
 tid=Int32((i-1)%step);blk=Int32((i-1)÷step)
 claim_site!(tid,blk,exact,cols,x,ns,seed);JACC.sync_workgroup();sample_scatter!(tid,blk,step,exact,cols,nbg,x,bg,obs,data);nothing
end
function specification(threads,blocks)
 JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=threads,blocks=blocks,shmem_size=0,sync=false)
end
function launch(a)
 threads=Int(min(Int32(256),a.cols));step=Int32(threads)
 if a.exact>0;JACC.parallel_for(exact_kernel!,specification(threads,Int(a.exact)),Int(a.exact)*threads,a.cols,a.nbg,a.x,a.bg,a.obs,a.data,step);end
 if a.sampled>0
  blocks=Int(a.sampled÷Int32(2));JACC.parallel_for(sampled_kernel!,specification(threads,blocks),blocks*threads,a.exact,a.cols,a.nbg,a.x,a.bg,a.obs,a.data,a.ns,a.seed,step)
 end
end
run_study(launch)
