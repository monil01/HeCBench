using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function compression!(dim,src,dst,cuts,offsets)
    tid=Int32(@index(Local,Linear)-1);global0=Int32(@index(Global,Linear)-1)
    scratch=@localmem Int32 (1536,)
    fpdc_lane!(dim,src,dst,cuts,offsets,scratch,tid,global0)
end
function factory(src,dst,cuts,offsets)
    kernel=compression!(CUDABackend(),64)
    dim->kernel(dim,src,dst,cuts,offsets;ndrange=256)
end
fpdc_study(factory)
