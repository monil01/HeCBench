using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function gru_ka!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 tx=Int32(@index(Local,Linear))-Int32(1);group=@index(Group,NTuple)
 gru_body!(tx,Int32(group[1])-Int32(1),Int32(group[2])-Int32(1),x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
end
const BACKEND=CUDABackend()
function run_layer!(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 gru_ka!(BACKEND,(128,1))(x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output;ndrange=(Int(B)*128,cld(Int(H),4)))
 nothing
end
study(run_layer!)
