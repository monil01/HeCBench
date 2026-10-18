using CUDA
include("../qt_device.jl")
function qt_kernel!(op,x,pc,N,D,threshold,blocks,seed)
 qt_site!(op,x,pc,N,D,threshold,blocks,seed)
 nothing
end
function launch(op,x,pc,N,D,threshold,blocks,seed,threads)
 grid=op in (:degree,:qtc) ? blocks : 1
 @cuda threads=threads blocks=grid qt_kernel!(Val(op),x,pc,N,D,threshold,blocks,seed)
 nothing
end
include("../study_common.jl")
qt_study(launch)
