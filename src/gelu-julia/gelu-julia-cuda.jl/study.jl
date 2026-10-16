using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(src,bias,mode,width,height,batch)
    if mode=="base"
        return ()->(@cuda threads=128 blocks=(height,batch) gelu_bias_loop_base!(src,bias,Int32(width),Int32(height)))
    end
    return ()->(@cuda threads=128 blocks=(height,batch) gelu_bias_loop_vec2!(src,bias,Int32(width),Int32(height)))
end
gelu_study(factory)
