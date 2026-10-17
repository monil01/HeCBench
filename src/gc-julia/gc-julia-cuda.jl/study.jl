include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare()
 dev=CUDA.device();sms=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);m=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
 (CuArray(IDX),CuArray(LIST),CUDA.fill(Int32(-1),N),CUDA.zeros(Int32,1),Int(sms*m÷THREADS_PER_BLOCK))
end
function execute(s)
 CUDA.fill!(s[3],Int32(-1));iter=0
 while true
  iter+=1;CUDA.fill!(s[4],Int32(0))
  @cuda blocks=s[5] threads=THREADS_PER_BLOCK color_kernel!(s[1],s[2],s[3],s[4],Int32(N))
  @cuda blocks=s[5] threads=THREADS_PER_BLOCK repair_conflicts_kernel!(s[1],s[2],s[3],s[4],Int32(N))
  CUDA.synchronize();changed=CUDA.@allowscalar s[4][1]
  if changed==0||iter>=1024;break;end
 end
 iter
end
study_main("cuda_jl",prepare,execute,s->s[3])
