using CUDA
include(joinpath(@__DIR__,"..","pointwise.jl"))
function cuda_entry!(tag,args,work)
 i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)-1
 stride=Int(blockDim().x)*Int(gridDim().x)
 while i<work
  point!(tag,i,args...)
  i+=stride
 end
 return
end
function launch(tag::Val,work::Int,args...)
 @cuda threads=128 blocks=128 cuda_entry!(tag,args,work)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;study_main("cuda_jl");end
