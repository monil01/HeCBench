using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference(")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,dense,output,offsets,mode)
    kernel=mode=="k1" ? dense_esuhm_kernel! : mode=="k2" ? dense_esuhm2_kernel! : dense_esuhm3_kernel!
    return ()->(@cuda threads=256 blocks=16 kernel(input,dense,output,Int32(768),offsets))
end
dense_study(factory)
