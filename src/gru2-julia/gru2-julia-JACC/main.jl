using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function gru_jacc!(i,j,x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 tx=mod(Int32(i)-Int32(1),Int32(128));b=(Int32(i)-Int32(1))÷Int32(128);tile=Int32(j)-Int32(1)
 gru_body!(tx,b,tile,x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 nothing
end
function run_layer!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 tiles=cld(Int(H),4);spec=JACC.launch_spec(threads=(128,1),blocks=(Int(B),tiles),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(Int(B)*128,tiles),gru_jacc!,x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 nothing
end
study(run_layer!)
