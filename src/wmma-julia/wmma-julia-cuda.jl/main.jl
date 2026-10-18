using CUDA
include("../study_device.jl")
function gemm_impl0(A,B,C,D,n,k)
 row=Int32(blockIdx().x-1)*Int32(16);col=Int32(blockIdx().y-1)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(0));nothing
end
function gemm_impl1(A,B,C,D,n,k)
 row=Int32(blockIdx().x-1)*Int32(64)+Int32((threadIdx().x-1)÷32)*Int32(16)
 col=Int32(blockIdx().y-1)*Int32(64)+Int32(threadIdx().y-1)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(1));nothing
end
function factory(A,B,C,D,c)
 if c.impl==0
  ()->(@cuda threads=32 blocks=(c.m÷16,c.n÷16) gemm_impl0(A,B,C,D,Int32(c.n),Int32(c.k)))
 else
  ()->(@cuda threads=(128,4) blocks=(c.m÷64,c.n÷64) gemm_impl1(A,B,C,D,Int32(c.n),Int32(c.k)))
 end
end
include("../study_common.jl")
wmma_study(CuArray,factory)
