using CUDA
include(joinpath(@__DIR__,"..","sigmoid_math.jl"))
function sigmoid_loss_kernel!(out,logits,targets,inner::Int32,mode::Int32)
 tid=threadIdx().x-Int32(1);row=blockIdx().x-Int32(1);last=(row+Int32(1))*inner
 value=0f0
 @inbounds for i in row*inner+tid:Int32(256):last-Int32(1)
  value+=sigmoid_term(logits[i+Int32(1)],targets[i+Int32(1)],mode)
 end
 pool=@cuStaticSharedMem(Float32,256);pool[tid+Int32(1)]=value;sync_threads()
 step=Int32(128)
 while step>0
  if tid<step;@inbounds pool[tid+Int32(1)]+=pool[tid+step+Int32(1)];end
  sync_threads();step>>=1
 end
 if tid==0;@inbounds out[row+Int32(1)]=-pool[1]/Float32(inner);end
 return
end
function study_launch!(d,mode)
 @cuda threads=256 blocks=Int(d[4]) sigmoid_loss_kernel!(d[3],d[1],d[2],d[5],mode)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==(@__FILE__);study_main("cuda_jl",study_launch!);end
