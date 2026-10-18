using CUDA
include("../study_device.jl")
include("../study_common.jl")
function compression!(dim,src,dst,cuts,offsets)
    tid=Int32(threadIdx().x-1);global0=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    scratch=CUDA.CuStaticSharedArray(Int32,1536)
    fpdc_lane!(dim,src,dst,cuts,offsets,scratch,tid,global0)
    nothing
end
function factory(src,dst,cuts,offsets)
    dim->CUDA.@cuda(threads=64,blocks=4,compression!(dim,src,dst,cuts,offsets))
end
fpdc_study(factory)
