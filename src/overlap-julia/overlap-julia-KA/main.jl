using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_inc!(out,input,n::Int32,inner::Int32)
    idx=Int32(@index(Global,Linear))-Int32(1)
    if idx<n
        @inbounds for j in Int32(0):inner-Int32(1)
            out[idx+Int32(1)]=(j==Int32(0) ? input[idx+Int32(1)] : out[idx+Int32(1)])+Int32(1)
        end
    end
    nothing
end
function study_prepare(din,dout,n,inner)
    kernel=portable_inc!(CUDABackend(),256);s->kernel(dout[s],din[s],Int32(n),inner;ndrange=n)
end
exit(study_main(study_prepare))
