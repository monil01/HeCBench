using CUDA,JACC
include("../qt_device.jl")
function qt_kernel!(i,op,x,pc,N,D,threshold,blocks,seed)
 qt_site!(op,x,pc,N,D,threshold,blocks,seed)
 nothing
end
function launch(op,x,pc,N,D,threshold,blocks,seed,threads)
 grid=op in (:degree,:qtc) ? blocks : 1
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=threads,blocks=grid,shmem_size=0,sync=false)
 JACC.parallel_for(qt_kernel!,spec,grid*threads,Val(op),x,pc,N,D,threshold,blocks,seed)
 nothing
end
include("../study_common.jl")
qt_study(launch)
