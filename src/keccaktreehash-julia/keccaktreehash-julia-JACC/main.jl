using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_permutation.jl"))
function tree_jacc!(i,input,output,constants)
 tid=(i-1)%64;bid=(i-1)÷64;state=leaf_state(input,constants,tid,bid)
 for w in 0:7;@inbounds output[tid+w*64+bid*512+1]=state[w+1];end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
launch(input,output,constants)=JACC.parallel_for(JACC.launch_spec(threads=64,blocks=3,shmem_size=0,sync=false),192,tree_jacc!,input,output,constants)
study(launch,(T,n)->JACC.zeros(T,n))
