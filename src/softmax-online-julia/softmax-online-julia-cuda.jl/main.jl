using CUDA
function online_softmax_kernel!(out,inp,nrows::Int32,cols::Int32)
 row=blockIdx().x-Int32(1);tid=threadIdx().x-Int32(1);width=blockDim().x
 smem=@cuStaticSharedMem(Float32,1024)
 maxval=-Inf32;sumval=0f0
 @inbounds for i in tid:width:cols-Int32(1)
  v=inp[row*cols+i+Int32(1)]
  if v>maxval;sumval*=exp(maxval-v);maxval=v;end
  sumval+=exp(v-maxval)
 end
 @inbounds smem[tid+Int32(1)]=maxval
 sync_threads();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]=max(smem[tid+Int32(1)],smem[tid+stride+Int32(1)]);end
  sync_threads();stride÷=Int32(2)
 end
 row_max=smem[1];sync_threads()
 @inbounds smem[tid+Int32(1)]=sumval*exp(maxval-row_max)
 sync_threads();stride=width÷Int32(2)
 while stride>0
  if tid<stride;@inbounds smem[tid+Int32(1)]+=smem[tid+stride+Int32(1)];end
  sync_threads();stride÷=Int32(2)
 end
 row_sum=smem[1];sync_threads()
 @inbounds for i in tid:width:cols-Int32(1)
  out[row*cols+i+Int32(1)]=exp(inp[row*cols+i+Int32(1)]-row_max)/row_sum
 end
 return
end
function study_launch!(f,w)
 @cuda threads=w blocks=Int(f[3]) online_softmax_kernel!(f[2],f[1],f[3],f[4])
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==(@__FILE__);study_main("cuda_jl",study_launch!);end
