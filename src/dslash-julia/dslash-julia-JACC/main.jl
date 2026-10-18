using CUDA,JACC
include(joinpath(@__DIR__,"..","dslash_math.jl"))
function back_kernel!(i,args...)
 back_site!(i-1,args...);nothing
end
function hopping_kernel!(i,args...)
 dslash_site!(i-1,args...);nothing
end
const SPECS=Dict{Tuple{Int,Int},Any}()
function launch_back(model,args...)
 n=Int(args[end]);spec=get!(SPECS,(n,1)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=1,blocks=n,shmem_size=0,sync=false)
 end
 JACC.parallel_for(back_kernel!,spec,n,args...)
end
function launch_dslash(model,args...)
 n=cld(Int(args[end]),256)*256;spec=get!(SPECS,(n,256)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=n÷256,shmem_size=0,sync=false)
 end
 JACC.parallel_for(hopping_kernel!,spec,n,args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:jacc)
