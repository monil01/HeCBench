
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline swap_value!(x,i)=reinterpret(Float32,CUDA.atomic_xchg!(reinterpret(Core.LLVMPtr{UInt32,1},pointer(x,i)),reinterpret(UInt32,1f0)))
function exact_kernel!(cols,nbg,x,bg,obs,data)
 tid=Int32(threadIdx().x)-Int32(1);row=Int32(blockIdx().x)-Int32(1);step=Int32(blockDim().x);exact_site!(tid,row,step,cols,nbg,x,bg,obs,data);return
end
function sampled_kernel!(exact,cols,nbg,x,bg,obs,data,ns,seed)
 tid=Int32(threadIdx().x)-Int32(1);blk=Int32(blockIdx().x)-Int32(1);step=Int32(blockDim().x)
 claim_site!(tid,blk,exact,cols,x,ns,seed);sync_threads();sample_scatter!(tid,blk,step,exact,cols,nbg,x,bg,obs,data);return
end
function launch(a)
 threads=Int(min(Int32(256),a.cols))
 if a.exact>0;@cuda threads=threads blocks=Int(a.exact) exact_kernel!(a.cols,a.nbg,a.x,a.bg,a.obs,a.data);end
 if a.sampled>0;@cuda threads=threads blocks=Int(a.sampled÷Int32(2)) sampled_kernel!(a.exact,a.cols,a.nbg,a.x,a.bg,a.obs,a.data,a.ns,a.seed);end
end
run_study(launch)
