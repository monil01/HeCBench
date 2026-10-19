using CUDA,Printf
include("../thermal_device.jl")
function thermal_cuda!(a,p,b,n,z,q)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);j=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
 column!(a,p,b,i,j,n,z,q)
 return
end
function factory(n)
 return (a,p,b,z,q)->(@cuda threads=(64,4) blocks=(div(n,64),div(n,4)) thermal_cuda!(a,p,b,Int32(n),Int32(z),q))
end
include("../study_common.jl")
thermal_study(factory)
