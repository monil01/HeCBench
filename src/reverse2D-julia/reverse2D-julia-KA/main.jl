using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function reverse_ka!(input,out,nr::Int32,nc::Int32,rm::Bool,ar::Bool,offset::Int32)
 i=@index(Global,Linear)-1
 if i<nr*nc
  row=rm ? i÷nc : i%nr;col=rm ? i%nc : i÷nr
  sr=ar ? nr-row-1 : row;sc=ar ? col : nc-col-1;j=rm ? sr*nc+sc : sc*nr+sr
  @inbounds out[offset+i+1]=input[offset+j+1]
 end
end
function factory(input,out,nr,nc,rm,ar,offset)
 kernel=reverse_ka!(CUDABackend(),256)
 ()->kernel(input,out,nr,nc,rm,ar,offset;ndrange=nr*nc)
end
reverse_study(factory)
