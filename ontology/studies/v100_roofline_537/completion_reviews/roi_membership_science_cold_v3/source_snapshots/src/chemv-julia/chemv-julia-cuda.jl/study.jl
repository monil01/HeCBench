include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main())"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(at,x,y)
return ()->begin
@cuda threads=32 blocks=12 chemv_kernel0!(at,x,y,1.59f0,3.14f0,8.28f0,2.71f0)
@cuda threads=32 blocks=12 chemv_kernel1!(at,x,y,1.59f0,3.14f0)
end
end
chemv_study(factory)
