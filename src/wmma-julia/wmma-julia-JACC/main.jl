using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
function gemm_impl0(i,j,A,B,C,D,n,k)
 row=Int32((i-1)÷32)*Int32(16);col=Int32(j-1)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(0));nothing
end
function gemm_impl1(i,j,A,B,C,D,n,k)
 row=Int32((i-1)÷128)*Int32(64)+Int32(((i-1)%128)÷32)*Int32(16)
 col=Int32((j-1)÷4)*Int32(64)+Int32((j-1)%4)*Int32(16)
 wmma_tile(A,B,C,D,n,k,row,col,Val(1));nothing
end
function makearray(a)
 d=JACC.array(a);d isa CUDA.CuArray||error("JACCmustallocateCUDAdevicearray");d
end
function factory(A,B,C,D,c)
 if c.impl==0
  spec=JACC.launch_spec(;threads=(32,1),blocks=(c.m÷16,c.n÷16),shmem_size=0,sync=false)
  ()->JACC.parallel_for(spec,(c.m÷16*32,c.n÷16),gemm_impl0,A,B,C,D,Int32(c.n),Int32(c.k))
 else
  spec=JACC.launch_spec(;threads=(128,4),blocks=(c.m÷64,c.n÷64),shmem_size=0,sync=false)
  ()->JACC.parallel_for(spec,(c.m÷64*128,c.n÷64*4),gemm_impl1,A,B,C,D,Int32(c.n),Int32(c.k))
 end
end
include("../study_common.jl")
wmma_study(makearray,factory)
