using CUDA,KernelAbstractions
include("../qt_device.jl")
@kernel unsafe_indices=true function qt_kernel!(op,x,pc,N,D,threshold,blocks,seed)
 qt_site!(op,x,pc,N,D,threshold,blocks,seed)
end
const FULL=qt_kernel!(CUDABackend(),64)
const ONE=qt_kernel!(CUDABackend(),1)
function launch(op,x,pc,N,D,threshold,blocks,seed,threads)
 grid=op in (:degree,:qtc) ? blocks : 1
 k=threads==1 ? ONE : FULL
 k(Val(op),x,pc,N,D,threshold,blocks,seed;ndrange=grid*threads)
 nothing
end
include("../study_common.jl")
qt_study(launch)
