using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(input,output,pad,offset)
 id=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)+offset;encode_body(id,input,output,pad);return
end
run!(input,output,pad,offset,n,stream)=(@cuda threads=256 blocks=cld(n,256) stream=stream kernel!(input,output,pad,offset);nothing)
study(run!)
