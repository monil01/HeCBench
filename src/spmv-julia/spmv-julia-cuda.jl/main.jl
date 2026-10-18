using CUDA
include("../study_device.jl")
function dense_cuda!(a,x,y,n)
 i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)
 dense_body!(a,x,y,n,i)
 return nothing
end
launch(a,x,y,n)=(@cuda threads=256 blocks=cld(n,256) dense_cuda!(a,x,y,n))
include("../study_common.jl")
run_study(launch)
