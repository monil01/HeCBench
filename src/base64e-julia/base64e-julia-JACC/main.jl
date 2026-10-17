using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
kernel!(id,input,output,pad,offset)=encode_body(id-Int32(1)+offset,input,output,pad)
function run!(input,output,pad,offset,n,stream)
 spec=JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false,stream=stream)
 JACC.parallel_for(spec,n,kernel!,input,output,pad,offset)
 return
end
study(run!)
