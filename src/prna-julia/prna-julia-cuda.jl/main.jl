using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function rna_init_kernel(c,n)
 c.w5[1]=c.w5[2]=c.w3[n]=c.w3[n+Int32(1)]=0f0
 return nothing
end
function rna_scalar_kernel(stage,c,n,d)
 i=Int32(blockIdx().x-1);rna_scalar_stage(stage,c,n,d,i)
 return nothing
end
function rna_partial_kernel(stage::Val{S},c,n,d) where S
 i=Int32(blockIdx().x-1);tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1)
 lane=tx;tid=S==2 ? tx*Int32(16)+ty : tx
 a,b,action=rna_partial(stage,c,n,d,i,tx,ty,lane)
 scratch=CUDA.CuStaticSharedArray(Float32,128)
 if action==2
  a=rna_source_reduce_cuda(a,tid,scratch)
  if S==6||S==8;b=rna_source_reduce_cuda(b,tid,scratch);end
 end
 if tid==0;rna_finish(stage,c,n,d,i,a,b,action);end
 return nothing
end
function launch_init(c,n)
 @cuda threads=1 blocks=1 rna_init_kernel(c,n)
 nothing
end
function launch_stage!(s::Val{S},c,n,d) where S
 if S==1||S==4||S==7
  @cuda threads=1 blocks=Int(n) rna_scalar_kernel(s,c,n,d)
 elseif S==2
  @cuda threads=(8,16) blocks=Int(n) rna_partial_kernel(s,c,n,d)
 elseif S==8
  @cuda threads=128 blocks=1 rna_partial_kernel(s,c,n,d)
 else
  @cuda threads=128 blocks=Int(n) rna_partial_kernel(s,c,n,d)
 end
 nothing
end
prna_study()
