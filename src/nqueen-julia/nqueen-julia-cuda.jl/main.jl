using CUDA
include("../study_common.jl")
function queen_kernel(roots,tree,solutions,P)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=P
  @inbounds t,s=queen_dfs(roots[i],Int32(15),Int32(7))
  @inbounds tree[i]=t;solutions[i]=s
 end
 return
end
function factory(roots,tree,solutions,P)
 ()->begin;@cuda threads=128 blocks=cld(Int(P),128) queen_kernel(roots,tree,solutions,P);end
end
main()=nqueen_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
