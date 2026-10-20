using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function rna_init_kernel(c,n)
 c.w5[1]=c.w5[2]=c.w3[n]=c.w3[n+Int32(1)]=0f0
end
@kernel function rna_scalar_kernel(stage,c,n,d)
 i=Int32(@index(Group,Linear)-1)
 rna_scalar_stage(stage,c,n,d,i)
end
@kernel function rna_partial_kernel(stage::Val{S},c,n,d) where S
 i=Int32(@index(Group,Linear)-1);lane=Int32(@index(Local,Linear)-1)
 tx=S==2 ? rem(lane,Int32(8)) : lane
 ty=S==2 ? div(lane,Int32(8)) : Int32(0)
 tid=S==2 ? tx*Int32(16)+ty : lane
 a,b,action=rna_partial(stage,c,n,d,i,tx,ty,lane)
 scratch=@localmem Float32 (128,)
 if action==2
  @inbounds scratch[tid+Int32(1)]=a
  @synchronize
  stride=Int32(64)
  while stride>0
   if tid<stride
    @inbounds scratch[tid+Int32(1)]=free_energy_sum(scratch[tid+Int32(1)],scratch[tid+stride+Int32(1)])
   end
   @synchronize
   stride=div(stride,Int32(2))
  end
  a=tid==0 ? (@inbounds scratch[1]) : a
  @synchronize
  if S==6||S==8
   @inbounds scratch[tid+Int32(1)]=b
   @synchronize
   stride2=Int32(64)
   while stride2>0
    if tid<stride2
     @inbounds scratch[tid+Int32(1)]=free_energy_sum(scratch[tid+Int32(1)],scratch[tid+stride2+Int32(1)])
    end
    @synchronize
    stride2=div(stride2,Int32(2))
   end
   b=tid==0 ? (@inbounds scratch[1]) : b
   @synchronize
  end
 end
 if tid==0;rna_finish(stage,c,n,d,i,a,b,action);end
end
function launch_init(c,n)
 backend=KernelAbstractions.get_backend(c.seq)
 rna_init_kernel(backend,1)(c,n;ndrange=1)
 nothing
end
function launch_stage!(s::Val{S},c,n,d) where S
 backend=KernelAbstractions.get_backend(c.seq)
 if S==1||S==4||S==7
  rna_scalar_kernel(backend,1)(s,c,n,d;ndrange=Int(n))
 else
  blocks=S==8 ? 1 : Int(n)
  rna_partial_kernel(backend,128)(s,c,n,d;ndrange=128*blocks)
 end
 nothing
end
prna_study()
