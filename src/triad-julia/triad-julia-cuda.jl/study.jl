source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"\nmain()\n"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_ref!(a,b,c,s,n)
    @cuda threads=128 blocks=cld(n,128) triad_kernel!(a,b,c,s)
end
triad_study(launch_ref!)
