using CUDA
include("../study_device.jl")
for op=0:8
 @eval function $(Symbol(:entry_,op))(args...)
  snicit_body(Val($op),args...)
  nothing
 end
end
function launch(op,args,a,stream)
 f=getfield(@__MODULE__,Symbol(:entry_,op))
 @cuda maxregs=64 threads=(Int(a[6]),Int(a[7])) blocks=Int(a[5]) shmem=Int(a[8]) stream=stream f(args...)
end
include("../study_common.jl")
main()
