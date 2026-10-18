using CUDA,JACC
include(joinpath(@__DIR__,"..","flame_math.jl"))
function initialize_kernel!(i,args...)
 initialize_site!(Int32(i-1),args...);nothing
end
function iterate_kernel!(i,args...)
 iterate_site!(Int32(i-1),args...);nothing
end
function generate_kernel!(i,args...)
 generate_site!(Int32(i-1),args...);nothing
end
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=32,blocks=8,shmem_size=0,sync=false)
function launch_initialize(model,args...)
 JACC.parallel_for(initialize_kernel!,SPEC,256,args...)
end
function launch_iterate(model,args...)
 JACC.parallel_for(iterate_kernel!,SPEC,256,args...)
end
function launch_generate(model,args...)
 JACC.parallel_for(generate_kernel!,SPEC,256,args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:jacc)
