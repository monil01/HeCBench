using CUDA
include("../study_device.jl")
for op=0:4
 @eval function $(Symbol(:entry_,op))(args...)
  width=UInt32(CUDA.blockDim().x);group=UInt32(CUDA.blockIdx().x-1);gid=group*width+UInt32(CUDA.threadIdx().x-1)
  ans_body(Val($op),args...,gid,group,width)
  nothing
 end
end
function launch(op,args,a)
 f=getfield(@__MODULE__,Symbol(:entry_,op))
 @cuda threads=128 blocks=Int(a[2]) stream=CUDA.legacy_stream() f(args...)
 nothing
end
include("../study_common.jl")
main()
