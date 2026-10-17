include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(x,y,z,mass,vx,vy,vz,n1,n2,cut,soft,fc)
 return ()->(@cuda threads=256 blocks=4 haccmk_kernel!(Int32(n1),Int32(n2),x,y,z,mass,vx,vy,vz,cut,soft,fc))
end
hacc_study(factory)
