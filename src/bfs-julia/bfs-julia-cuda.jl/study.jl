include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function expand!(s,n,e,m,u,v,c,N)
 @cuda threads=256 blocks=cld(N,256) bfs_expand!(s,n,e,m,u,v,c,Int32(N))
end
function update!(m,u,v,o,N)
 @cuda threads=256 blocks=cld(N,256) bfs_update!(m,u,v,o,Int32(N))
end
study_main(expand!,update!)
