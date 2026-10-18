using CUDA,StaticArrays
function inplace_padding!(a,flags,n,m,p,nt,cut)
 tid=Int32(threadIdx().x-1);task=cut+Int32(blockIdx().x-1);sz=m*(n+p)
 while task<nt
  reg=MVector{32,Float64}(undef);pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   row=pos÷(n+p);x=pos%(n+p);src=row*n+x
   reg[j]=src>=0 && x<n && src<sz ? a[src+Int32(1)] : 0.
   pos-=Int32(256)
  end
  sync_threads()
  if tid==0
   while CUDA.atomic_add!(pointer(flags,task+Int32(1)),Int32(0))==0;end
   CUDA.atomic_add!(pointer(flags,task+Int32(2)),Int32(1))
  end
  sync_threads();pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   if 0<=pos<sz;a[pos+Int32(1)]=reg[j];end
   pos-=Int32(256)
  end
  task+=Int32(8)
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
pad_study((a,f,n,m,p,nt,cut)->(@cuda threads=256 blocks=8 inplace_padding!(a,f,n,m,p,nt,cut)))
