source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_reference!(out,x,y,lengths,offsets,nt,chunk,blocks)
    @cuda threads=512 blocks=blocks axpby_kernel!(out,x,y,lengths,offsets,nt,chunk,1f0,1f0)
end
axpby_study(launch_reference!)
