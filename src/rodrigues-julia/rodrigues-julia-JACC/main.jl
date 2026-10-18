using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function rotate_jacc!(index,d,p,n::Int32,width)
 i=Int32(index)
 if i<=n;@inbounds rotate_point!(d,i,p[5],p[6],p[7],p[8],width);end
 return
end
function factory(d,p,width)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 n->JACC.parallel_for(spec,n,rotate_jacc!,d,p,n,width)
end
rodrigues_study(factory)
