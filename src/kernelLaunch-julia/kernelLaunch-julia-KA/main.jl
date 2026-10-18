using CUDA,KernelAbstractions
include("../study_types.jl")
@kernel function primary(args,out)
 i=@index(Global,Linear)
 kl_store(args,out,i-1)
end
@kernel function echo_payload(args::KLArgs{N},out) where N
 i=@index(Global,Linear)
 if i<=N
  @inbounds out[i]=args.bytes[Int(i)]
 end
end
const primary_object=primary(CUDABackend(),1)
const echo_object=echo_payload(CUDABackend(),256)
launch(args,out)=primary_object(args,out;ndrange=1)
launch_echo(args::KLArgs{N},out) where N=echo_object(args,out;ndrange=N)
include("../study_common.jl")
run_study(launch,launch_echo)
