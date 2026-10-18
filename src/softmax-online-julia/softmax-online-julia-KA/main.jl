using CUDA,KernelAbstractions
@kernel unsafe_indices=true function online_softmax_kernel!(out,inp,nrows::Int32,cols::Int32,width::Int32)
 row=Int32(@index(Group,Linear))-Int32(1);tid=Int32(@index(Local,Linear))-Int32(1)
 smem=@localmem(Float32,(1024,))
 maxval=-Inf32;sumval=0f0
 @inbounds for i in tid:width:cols-Int32(1)
  v=inp[row*cols+i+Int32(1)]
  if v>maxval;sumval*=exp(maxval-v);maxval=v;end
  sumval+=exp(v-maxval)
 end
 @inbounds smem[tid+Int32(1)]=maxval
 @synchronize();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]=max(smem[tid+Int32(1)],smem[tid+stride+Int32(1)]);end
  @synchronize();stride÷=Int32(2)
 end
 row_max=smem[1];@synchronize()
 @inbounds smem[tid+Int32(1)]=sumval*exp(maxval-row_max)
 @synchronize();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]+=smem[tid+stride+Int32(1)];end
  @synchronize();stride÷=Int32(2)
 end
 row_sum=smem[1];@synchronize()
 @inbounds for i in tid:width:cols-Int32(1)
  out[row*cols+i+Int32(1)]=exp(inp[row*cols+i+Int32(1)]-row_max)/row_sum
 end

end
const ST_KERNELS=Dict(w=>online_softmax_kernel!(CUDABackend(),w) for w in (32,64,128,256,512,1024))
function study_launch!(f,w)
 ST_KERNELS[w](f[2],f[1],f[3],f[4],Int32(w);ndrange=Int(f[3])*w)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("ka",study_launch!)
