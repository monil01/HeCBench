using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function kernel(x,w,y)
 g=@index(Global,Linear)
 if g<=512;lstm_lane!(x,w,y,g);end
end
clink_study((x,w,y)->begin;k=kernel(CUDABackend(),256);()->k(x,w,y;ndrange=512);end)
