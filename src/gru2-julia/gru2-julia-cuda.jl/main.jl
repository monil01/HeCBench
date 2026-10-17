include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function gru_ref_kernel!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 tx=threadIdx().x-Int32(1);b=blockIdx().x-Int32(1);tile=blockIdx().y-Int32(1)
 gru_body!(tx,b,tile,x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 return
end
function run_layer!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 @cuda threads=128 blocks=(B,cld(H,Int32(4))) gru_ref_kernel!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 nothing
end
study(run_layer!)
