using CUDA
include("../study_types.jl")
function primary(args,out)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 kl_store(args,out,i)
 return
end
function echo_payload(args::KLArgs{N},out) where N
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=N
  @inbounds out[i]=args.bytes[Int(i)]
 end
 return
end
launch(args,out)=(@cuda threads=1 blocks=1 primary(args,out))
launch_echo(args::KLArgs{N},out) where N=(@cuda threads=256 blocks=cld(N,256) echo_payload(args,out))
include("../study_common.jl")
run_study(launch,launch_echo)
