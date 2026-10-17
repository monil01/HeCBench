using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,a,b,c,n)
 add_body(Int32(i)-Int32(1),a,b,c,n);return
end
run!(a,b,c,n)=(JACC.parallel_for(JACC.launch_spec(threads=256,shmem_size=0,sync=false),Int(n),kernel!,a,b,c,n);nothing)
study(run!)
