using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,gate,voltage,n)
 if i<=n
 @inbounds gate[i]=step_gate(gate[i],voltage[i])
 end
 return
end
run!(g,v,n)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,kernel!,g,v,Int32(n));nothing)
study(run!)
