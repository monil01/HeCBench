using CUDA,JACC,Printf
JACC.@init_backend
include("../study_device.jl")
function quant_kernel(i,A,B,C,book,m::Int32,k::Int32,bvec::Int32,cvec::Int32,nt::Int32)
 tx=Int32((i-1)%nt);lane=tx%Int32(32);row=Int32((i-1)÷nt)*(nt÷Int32(32))+tx÷Int32(32)
 sh=CUDA.CuStaticSharedArray(Float16,2304);res=0f0
 for tile=Int32(0):Int32(cld(k÷8,256)-1)
  CUDA.sync_threads();q_load!(sh,B,bvec,tile*256,tx,nt,k);CUDA.sync_threads()
  res=q_accum(sh,A,book,row,lane,tile,k,m,res)
 end
 q_reduce_store!(C,cvec,row,lane,m,res)
 nothing
end
function factory(a,m,k,blocks,threads)
 A,B,C,book=a;spec=JACC.launch_spec(;threads=threads,blocks=blocks,shmem_size=4608,sync=false)
 (bvec,cvec)->JACC.parallel_for(spec,blocks*threads,quant_kernel,A,B,C,book,Int32(m),Int32(k),Int32(bvec),Int32(cvec),Int32(threads))
end
include("../study_common.jl")
quant_study(factory)
