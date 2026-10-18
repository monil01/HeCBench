using CUDA,JACC
function online_softmax_callback(i,out,inp,nrows::Int32,cols::Int32,width::Int32,shift::Int32,template)
 gid=Int32(i)-Int32(1);row=gid>>shift;tid=gid&(width-Int32(1))
 smem=JACC.shared(template)
 maxval=-Inf32;sumval=0f0
 @inbounds for i in tid:width:cols-Int32(1)
  v=inp[row*cols+i+Int32(1)]
  if v>maxval;sumval*=exp(maxval-v);maxval=v;end
  sumval+=exp(v-maxval)
 end
 @inbounds smem[tid+Int32(1)]=maxval
 JACC.sync_workgroup();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]=max(smem[tid+Int32(1)],smem[tid+stride+Int32(1)]);end
  JACC.sync_workgroup();stride÷=Int32(2)
 end
 row_max=smem[1];JACC.sync_workgroup()
 @inbounds smem[tid+Int32(1)]=sumval*exp(maxval-row_max)
 JACC.sync_workgroup();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]+=smem[tid+stride+Int32(1)];end
  JACC.sync_workgroup();stride÷=Int32(2)
 end
 row_sum=smem[1];JACC.sync_workgroup()
 @inbounds for i in tid:width:cols-Int32(1)
  out[row*cols+i+Int32(1)]=exp(inp[row*cols+i+Int32(1)]-row_max)/row_sum
 end
 return
end
const BACKEND=JACC.get_backend(:cuda)
const SPECS=Dict((n,w)=>JACC.LaunchSpec{typeof(BACKEND)}(;threads=w,blocks=n,shmem_size=4096,sync=false) for n in (17,19,33,3) for w in (32,64,128,256,512,1024))
const SHIFTS=Dict(w=>Int32(trailing_zeros(UInt32(w))) for w in (32,64,128,256,512,1024))
const TEMPLATE=CUDA.zeros(Float32,1024)
function study_launch!(f,w)
 JACC.parallel_for(online_softmax_callback,SPECS[(Int(f[3]),w)],Int(f[3])*w,f[2],f[1],f[3],f[4],Int32(w),SHIFTS[w],TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("jacc",study_launch!)
