using CUDA
include(joinpath(@__DIR__,"study_permutation.jl"))
function tree_cuda!(input,output,constants)
 tid=Int32(threadIdx().x-1);bid=Int32(blockIdx().x-1);state=leaf_state(input,constants,tid,bid)
 for w in 0:7;@inbounds output[tid+w*64+bid*512+1]=state[w+1];end
 return
end
