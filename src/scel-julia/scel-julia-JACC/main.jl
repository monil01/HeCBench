using CUDA,JACC
include(joinpath(@__DIR__,"..","sigmoid_math.jl"))
function sigmoid_loss_callback(i,out,logits,targets,inner::Int32,mode::Int32,template)
 tid=(Int32(i)-Int32(1))%Int32(256);row=(Int32(i)-Int32(1))÷Int32(256);last=(row+Int32(1))*inner
 value=0f0
 @inbounds for i in row*inner+tid:Int32(256):last-Int32(1)
  value+=sigmoid_term(logits[i+Int32(1)],targets[i+Int32(1)],mode)
 end
 pool=JACC.shared(template);pool[tid+Int32(1)]=value;JACC.sync_workgroup()
 step=Int32(128)
 while step>0
  if tid<step;@inbounds pool[tid+Int32(1)]+=pool[tid+step+Int32(1)];end
  JACC.sync_workgroup();step>>=1
 end
 if tid==0;@inbounds out[row+Int32(1)]=-pool[1]/Float32(inner);end
 return
end
const BACKEND=JACC.get_backend(:cuda)
const SPECS=Dict(n=>JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=n,shmem_size=1024,sync=false) for n in (17,33,65,9))
const TEMPLATE=CUDA.zeros(Float32,256)
function study_launch!(d,mode)
 JACC.parallel_for(sigmoid_loss_callback,SPECS[Int(d[4])],Int(d[4])*256,d[3],d[1],d[2],d[5],mode,TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("jacc",study_launch!)
