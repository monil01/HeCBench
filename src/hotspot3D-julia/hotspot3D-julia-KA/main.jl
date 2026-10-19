using CUDA,KernelAbstractions,Printf
include("../thermal_device.jl")
@kernel function thermal_ka!(a,p,b,n,z,q)
 ij=@index(Global,NTuple)
 column!(a,p,b,Int32(ij[1]-1),Int32(ij[2]-1),n,z,q)
end
function factory(n)
 return (a,p,b,z,q)->thermal_ka!(get_backend(a),(64,4))(a,p,b,Int32(n),Int32(z),q;ndrange=(n,n))
end
include("../study_common.jl")
thermal_study(factory)
