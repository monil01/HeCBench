using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
# Named study dimensions match native constants; preserved source remains intact.
source=replace(source,"const INSTANCES = 512"=>"const INSTANCES = 224","const ATTRIBUTES = 100"=>"const ATTRIBUTES = 4096","\nmain()\n"=>"\n")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(output,input,instances,attributes)
    return ()->(@cuda threads=1 blocks=(instances,instances) apd_kernel!(output,input))
end
apd_study(factory)
