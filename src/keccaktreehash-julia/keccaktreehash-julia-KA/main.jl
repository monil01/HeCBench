using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_permutation.jl"))
@kernel function tree_ka!(input,output,constants)
 tid=@index(Local,Linear);bid=@index(Group,Linear);state=leaf_state(input,constants,tid-1,bid-1)
 for w in 0:7;@inbounds output[tid+w*64+(bid-1)*512]=state[w+1];end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const TREE=tree_ka!(BACKEND,64)
launch(input,output,constants)=TREE(input,output,constants;ndrange=192)
study(launch,(T,n)->KernelAbstractions.allocate(BACKEND,T,n))
