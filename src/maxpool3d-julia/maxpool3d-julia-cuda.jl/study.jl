using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,output,w,h,count,ow,oh)
    return ()->(@cuda threads=(8,8,4) blocks=(cld(ow,8),cld(oh,8),cld(count,4)) maxpool3d!(input,output,Int32(2),Int32(2),Int32(2),Int32(2),count,w,h,ow,oh))
end
pool_study(factory)
