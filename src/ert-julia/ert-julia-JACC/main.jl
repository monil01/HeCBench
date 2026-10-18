using CUDA,JACC
include(joinpath(@__DIR__,"..","ert_math.jl"))
function ert_kernel!(i,A,n,trials,total)
 ert_site!(UInt32(i-1),total,A,n,trials);nothing
end
const SPECS=Dict{Tuple{Int,Int},Any}()
function launch_ert(model,A,n,t,blocks,threads)
 spec=get!(SPECS,(blocks,threads)) do
  JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=threads,blocks=blocks,shmem_size=0,sync=false)
 end
 JACC.parallel_for(ert_kernel!,spec,blocks*threads,A,UInt32(n),UInt32(t),UInt32(blocks*threads))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:jacc)
