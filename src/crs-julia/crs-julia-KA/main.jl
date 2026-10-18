using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
@kernel function coding_kernel!(::Val{W},::Val{M},k,data,code,bm,size) where {W,M}
 tid=Int32(@index(Local,Linear)-1);block=Int32(@index(Group,Linear)-1);work=Int32(div(128,W)*W);idx=work*block+tid;active=tid<work&&idx<size
 shared=@localmem(UInt64,(128,));r=@private(UInt64,(M,));
 @inbounds for ri in 1:M;r[ri]=UInt64(0);end
 group=div(tid,Int32(W))*Int32(W);inner=rem(tid,Int32(W))
 for ki in Int32(0):k-Int32(1)
  if active;@inbounds shared[tid+1]=data[ki*size+idx+1];end
  @synchronize
  if active
   @inbounds for j in 0:W-1
    mask=bm[ki*W+j+1];word=shared[group+j+1]
    for row in 0:M-1;r[row+1] ⊻= (UInt64(0)-UInt64((mask>>(inner+row*W))&UInt32(1)))&word;end
   end
  end
  @synchronize
 end
 if active;@inbounds for row in 0:M-1;code[row*size+idx+1]=r[row+1];end;end
end
function launch(k,m,w,data,code,bm,size,blocks)
 coding_kernel!(BACKEND,128)(Val(Int(w)),Val(Int(m)),k,data,code,bm,size;ndrange=blocks*128)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
