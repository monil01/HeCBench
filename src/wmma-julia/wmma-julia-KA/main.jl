using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function gemm_impl0(A,B,C,D,n,k)
 g=@index(Group,NTuple)
 row=Int32(g[1]-1)*Int32(16);col=Int32(g[2]-1)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(0))
end
@kernel function gemm_impl1(A,B,C,D,n,k)
 g=@index(Group,NTuple);l=@index(Local,NTuple)
 row=Int32(g[1]-1)*Int32(64)+Int32((l[1]-1)÷32)*Int32(16)
 col=Int32(g[2]-1)*Int32(64)+Int32(l[2]-1)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(1))
end
function makearray(a)
 d=KernelAbstractions.allocate(CUDABackend(),eltype(a),(length(a),));copyto!(d,a);d
end
function factory(A,B,C,D,c)
 if c.impl==0
  kernel=gemm_impl0(CUDABackend(),(32,1));()->kernel(A,B,C,D,Int32(c.n),Int32(c.k);ndrange=(c.m÷16*32,c.n÷16))
 else
  kernel=gemm_impl1(CUDABackend(),(128,4));()->kernel(A,B,C,D,Int32(c.n),Int32(c.k);ndrange=(c.m÷64*128,c.n÷64*4))
 end
end
include("../study_common.jl")
wmma_study(makearray,factory)
