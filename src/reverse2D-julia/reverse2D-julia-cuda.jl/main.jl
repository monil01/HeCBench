using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function reverse_cuda!(input,out,nr::Int32,nc::Int32,rm::Bool,ar::Bool,offset::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 if i<nr*nc
  row=rm ? i÷nc : i%nr;col=rm ? i%nc : i÷nr
  sr=ar ? nr-row-Int32(1) : row;sc=ar ? col : nc-col-Int32(1);j=rm ? sr*nc+sc : sc*nr+sr
  @inbounds out[offset+i+Int32(1)]=input[offset+j+Int32(1)]
 end
 return
end
function factory(input,out,nr,nc,rm,ar,offset)
 ()->(@cuda threads=256 blocks=cld(nr*nc,256) reverse_cuda!(input,out,nr,nc,rm,ar,offset))
end
reverse_study(factory)
