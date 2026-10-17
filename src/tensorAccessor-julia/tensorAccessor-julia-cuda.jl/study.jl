include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(mode,f,d)
 if mode==1
  @cuda threads=256 blocks=cld(f[1],256) raw_study!(Int64(f[1]),Int64(f[2]),d[3],d[1],d[2])
 else
  @cuda threads=256 blocks=cld(f[1],256) packed_study!(d[5],d[6],d[7])
 end
end
study_main(launch!)
