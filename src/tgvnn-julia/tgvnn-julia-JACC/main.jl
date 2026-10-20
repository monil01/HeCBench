using CUDA,JACC
include(joinpath(@__DIR__,"..","pointwise.jl"))
CUDA.stream!(CUDA.legacy_stream())
const TGV_JACC_SPEC=JACC.launch_spec(threads=128,blocks=128,shmem_size=0,sync=false,stream=CUDA.legacy_stream())
function jacc_callback!(index,tag,args,work)
 i=Int(index)-1
 while i<work
  point!(tag,i,args...)
  i+=16384
 end
end
function launch(tag::Val,work::Int,args...)
 JACC.parallel_for(TGV_JACC_SPEC,16384,jacc_callback!,tag,args,work)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;study_main("jacc");end
