using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","sigmoid_math.jl"))
@kernel unsafe_indices=true function sigmoid_loss_kernel!(out,logits,targets,inner::Int32,mode::Int32)
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1);last=(row+Int32(1))*inner
 value=0f0
 @inbounds for i in row*inner+tid:Int32(256):last-Int32(1)
  value+=sigmoid_term(logits[i+Int32(1)],targets[i+Int32(1)],mode)
 end
 pool=@localmem(Float32,(256,));pool[tid+Int32(1)]=value;@synchronize()
 step=Int32(128)
 while step>0
  if tid<step;@inbounds pool[tid+Int32(1)]+=pool[tid+step+Int32(1)];end
  @synchronize();step>>=1
 end
 if tid==0;@inbounds out[row+Int32(1)]=-pool[1]/Float32(inner);end

end
const ST_KERNEL=sigmoid_loss_kernel!(CUDABackend(),256)
function study_launch!(d,mode)
 ST_KERNEL(d[3],d[1],d[2],d[5],mode;ndrange=Int(d[4])*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("ka",study_launch!)
