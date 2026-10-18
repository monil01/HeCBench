using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function reverse_jacc!(index,input,out,nr::Int32,nc::Int32,rm::Bool,ar::Bool,offset::Int32)
 i=index-1
 if i<nr*nc
  row=rm ? i÷nc : i%nr;col=rm ? i%nc : i÷nr
  sr=ar ? nr-row-1 : row;sc=ar ? col : nc-col-1;j=rm ? sr*nc+sc : sc*nr+sr
  @inbounds out[offset+i+1]=input[offset+j+1]
 end
 return
end
function factory(input,out,nr,nc,rm,ar,offset)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,nr*nc,reverse_jacc!,input,out,nr,nc,rm,ar,offset)
end
reverse_study(factory)
