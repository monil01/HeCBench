using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function queen_kernel(roots,tree,solutions,P)
 i=@index(Global,Linear)
 if i<=P
  @inbounds t,s=queen_dfs(roots[i],Int32(15),Int32(7))
  @inbounds tree[i]=t;solutions[i]=s
 end
end
function factory(roots,tree,solutions,P)
 kernel=queen_kernel(CUDABackend(),128)
 ()->kernel(roots,tree,solutions,P;ndrange=cld(Int(P),128)*128)
end
main()=nqueen_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
