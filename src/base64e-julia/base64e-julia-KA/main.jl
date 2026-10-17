using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(input,output,pad,offset)
 id=@index(Global,Linear);encode_body(id-Int32(1)+offset,input,output,pad)
end
function run!(input,output,pad,offset,n,stream)
 CUDA.stream!(stream) do
  kernel!(get_backend(output),256)(input,output,pad,offset;ndrange=n)
 end
 return
end
study(run!)
