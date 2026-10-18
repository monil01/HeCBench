using CUDA,JACC
JACC.@init_backend
include("../study_types.jl")
function primary(i,args,out)
 kl_store(args,out,i-1)
 return
end
function echo_payload(i,args::KLArgs{N},out) where N
 if i<=N
  @inbounds out[i]=args.bytes[Int(i)]
 end
 return
end
const primary_spec=JACC.launch_spec(;threads=1,blocks=1,sync=false)
launch(args,out)=JACC.parallel_for(primary_spec,1,primary,args,out)
launch_echo(args::KLArgs{N},out) where N=JACC.parallel_for(JACC.launch_spec(;threads=256,blocks=cld(N,256),sync=false),N,echo_payload,args,out)
include("../study_common.jl")
run_study(launch,launch_echo)
