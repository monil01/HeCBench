using CUDA,KernelAbstractions,Printf
include("../study_device.jl")
@kernel unsafe_indices=true function quant_kernel(@Const(A),@Const(B),C,@Const(book),m::Int32,k::Int32,bvec::Int32,cvec::Int32)
 tid=@index(Local,Linear);group=@index(Group,Linear)
 tx=Int32(tid-1);nt=Int32(@groupsize()[1]);lane=tx%Int32(32);row=Int32(group-1)*(nt÷Int32(32))+tx÷Int32(32)
 sh=@localmem Float16 (2304,)
 res=0f0
 for tile=Int32(0):Int32(cld(k÷8,256)-1)
  @synchronize
  q_load!(sh,B,bvec,tile*256,tx,nt,k)
  @synchronize
  res=q_accum(sh,A,book,row,lane,tile,k,m,res)
 end
 q_reduce_store!(C,cvec,row,lane,m,res)
end
function factory(a,m,k,blocks,threads)
 A,B,C,book=a;kern=quant_kernel(CUDABackend(),threads)
 (bvec,cvec)->kern(A,B,C,book,Int32(m),Int32(k),Int32(bvec),Int32(cvec);ndrange=blocks*threads)
end
include("../study_common.jl")
quant_study(factory)
