include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function nw_launch!(a,ref,dim,p)
 cols=Int32(dim+1)
 for d in 2:2dim
  lo=max(1,d-dim);hi=min(dim,d-1)
  @cuda threads=256 blocks=cld(hi-lo+1,256) nw_diag_kernel!(a,ref,cols,Int32(p),Int32(d),Int32(lo),Int32(hi))
 end
end
study_main(nw_launch!)
