using CUDA,JACC,StaticArrays,Atomix
function inplace_padding!(i,a,flags,n,m,p,nt,cut)
 z=Int32(i)-Int32(1);tid=z%Int32(256);task=cut+z÷Int32(256);sz=m*(n+p)
 while task<nt
  reg=MVector{32,Float64}(undef);pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   row=pos÷(n+p);x=pos%(n+p);src=row*n+x
   reg[j]=src>=0 && x<n && src<sz ? a[src+Int32(1)] : 0.
   pos-=Int32(256)
  end
  JACC.sync_workgroup()
  if tid==0
   while (Atomix.@atomic :monotonic flags[task+Int32(1)]+=Int32(0))==0;end
   Atomix.@atomic :monotonic flags[task+Int32(2)]+=Int32(1)
  end
  JACC.sync_workgroup()
  pos=sz-Int32(1)-(task*Int32(8192)+tid)
  for j in 1:32
   if 0<=pos<sz;a[pos+Int32(1)]=reg[j];end
   pos-=Int32(256)
  end
  task+=Int32(8)
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const PADS=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=8,shmem_size=0,sync=false)
pad_study((a,f,n,m,p,nt,cut)->JACC.parallel_for(inplace_padding!,PADS,2048,a,f,n,m,p,nt,cut))
