using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
source=source[findfirst("function attention_kernel1!",source).start:findfirst("function verify_kernel!",source).start-1]
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(key,value,query,dot,es,score,output)
    return ()->begin
        fill!(es,0f0)
        @cuda threads=256 blocks=4 attention_kernel1!(key,query,dot,es,Int32(1024),Int32(64))
        @cuda threads=256 blocks=4 attention_kernel2!(es,dot,score,Int32(1024))
        @cuda threads=256 blocks=1 attention_kernel3!(score,value,output,Int32(1024),Int32(64))
    end
end
attention_study(factory)
