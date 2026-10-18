using CUDA,KernelAbstractions,StaticArrays,Atomix
@kernel unsafe_indices=true function inplace_padding!(a,flags,n,m,p,nt,cut)
 tid=Int32(@index(Local,Linear))-Int32(1);task=cut+Int32(@index(Group,Linear))-Int32(1);sz=m*(n+p)
 while task<nt
  reg=MVector{32,Float64}(undef);pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   row=pos÷(n+p);x=pos%(n+p);src=row*n+x
   reg[j]=src>=0 && x<n && src<sz ? a[src+Int32(1)] : 0.
   pos-=Int32(256)
  end
  @synchronize
  if tid==0
   while (Atomix.@atomic :monotonic flags[task+Int32(1)]+=Int32(0))==0;end
   Atomix.@atomic :monotonic flags[task+Int32(2)]+=Int32(1)
  end
  @synchronize
  pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   if 0<=pos<sz;a[pos+Int32(1)]=reg[j];end
   pos-=Int32(256)
  end
  task+=Int32(8)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const PADK=inplace_padding!(CUDABackend(),256)
pad_study((a,f,n,m,p,nt,cut)->PADK(a,f,n,m,p,nt,cut;ndrange=2048))
