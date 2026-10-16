using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main(ARGS))"=>"")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_original!(a,v,r,t,h,n)
    @cuda threads=256 blocks=cld(n,256) calculate_range_kernel!(a,v,r,t,h,n)
end
exit(projectile_study(launch_original!))
