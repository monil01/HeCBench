using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function iv_jacc!(index,src,dst,n::Int32,layout)
 i=Int32(index)
 if i<=n;iv_accumulate!(src,dst,i,layout);end
 return
end
function factory(src,dst)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 (n,layout)->JACC.parallel_for(spec,n,iv_jacc!,src,dst,n,layout)
end
interleave_study(factory)
