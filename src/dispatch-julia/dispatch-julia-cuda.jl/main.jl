using CUDA
include("../study_common.jl")
function empty_kernel()
 nothing
end
launch()=CUDA.@cuda(threads=1,blocks=1,empty_kernel())
dispatch_study(launch)
