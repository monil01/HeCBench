using CUDA
using StaticArrays
function coding_kernel!(::Val{W},::Val{M},k,data,code,bm,size) where {W,M}
 tid=Int32(threadIdx().x-1);block=Int32(blockIdx().x-1);work=Int32(div(128,W)*W);idx=work*block+tid;active=tid<work&&idx<size
 shared=CuDynamicSharedArray(UInt64,128);r=MVector{M,UInt64}(ntuple(_->UInt64(0),Val(M)));group=div(tid,Int32(W))*Int32(W);inner=rem(tid,Int32(W))
 for ki in Int32(0):k-Int32(1)
  if active;@inbounds shared[tid+1]=data[ki*size+idx+1];end
  sync_threads()
  if active
   @inbounds for j in 0:W-1
    mask=bm[ki*W+j+1];word=shared[group+j+1]
    for row in 0:M-1;r[row+1] ⊻= (UInt64(0)-UInt64((mask>>(inner+row*W))&UInt32(1)))&word;end
   end
  end
  sync_threads()
 end
 if active;@inbounds for row in 0:M-1;code[row*size+idx+1]=r[row+1];end;end
 return
end
function launch(k,m,w,data,code,bm,size,blocks)
 @cuda threads=128 blocks=blocks shmem=1024 coding_kernel!(Val(Int(w)),Val(Int(m)),k,data,code,bm,size)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
