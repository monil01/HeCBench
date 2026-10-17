include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(path,dst,src,n)
 @cuda threads=256 blocks=cld(n,256) encode_study!(dst,src,n,Int32((path-1)÷2),iseven(path))
end
study_main(launch!)
