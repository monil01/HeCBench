using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
ending=findlast("\nexit(main())",source);ending===nothing&&error("Missing originalmain guard")
include_string(Main,source[1:first(ending)-1],joinpath(@__DIR__,"main.jl"))
include("../study_common.jl")
function factory(input,output,weight,bias,args)
 blocks=cld(Int(args[1]),256)
 ()->CUDA.@cuda(threads=256,blocks=blocks,dwconv_kernel!(input,output,weight,bias,args...))
end
dwconv_study(factory)
