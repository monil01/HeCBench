using CUDA, KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function portable_dwconv!(input,output,weight,bias,args)
    linear=Int32(@index(Global,Linear)-1)
    dwconv_lane!(linear,input,output,weight,bias,args...)
end
function factory(input,output,weight,bias,args)
    kernel=portable_dwconv!(CUDABackend(),256)
    ()->kernel(input,output,weight,bias,args;ndrange=Int(args[1]))
end
dwconv_study(factory)
