using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function queen_kernel(i,roots,tree,solutions,P)
 if i<=P
  @inbounds t,s=queen_dfs(roots[i],Int32(15),Int32(7))
  @inbounds tree[i]=t;solutions[i]=s
 end
end
function factory(roots,tree,solutions,P)
 spec=JACC.launch_spec(;threads=128,blocks=cld(Int(P),128),shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,cld(Int(P),128)*128,queen_kernel,roots,tree,solutions,P)
end
main()=nqueen_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
