
using CUDA,Printf
include("../study_device.jl")
function quant_kernel(A,B,C,book,m::Int32,k::Int32,bvec::Int32,cvec::Int32)
 tx=Int32(threadIdx().x-1);nt=Int32(blockDim().x);lane=tx%Int32(32);row=Int32(blockIdx().x-1)*(nt÷Int32(32))+tx÷Int32(32)
 sh=CUDA.CuStaticSharedArray(Float16,2304);res=0f0
 for tile=Int32(0):Int32(cld(k÷8,256)-1)
  CUDA.sync_threads();q_load!(sh,B,bvec,tile*256,tx,nt,k);CUDA.sync_threads()
  res=q_accum(sh,A,book,row,lane,tile,k,m,res)
 end
 q_reduce_store!(C,cvec,row,lane,m,res)
 nothing
end
function factory(a,m,k,blocks,threads)
 A,B,C,book=a
 (bvec,cvec)->(@cuda threads=threads blocks=blocks shmem=4608 quant_kernel(A,B,C,book,Int32(m),Int32(k),Int32(bvec),Int32(cvec)))
end
include("../study_common.jl")
quant_study(factory)
