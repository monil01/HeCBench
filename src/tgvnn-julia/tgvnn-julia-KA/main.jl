using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","pointwise.jl"))
@kernel function ka_entry!(tag,args,work)
 i=@index(Global,Linear)-1
 while i<work
  point!(tag,i,args...)
  i+=16384
 end
end
function launch(tag::Val,work::Int,args...)
 kernel=ka_entry!(get_backend(args[1]),128)
 kernel(tag,args,work;ndrange=16384)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;study_main("ka");end
