using CUDA,KernelAbstractions
import Atomix
const MODEL="ka"
@inline function density_add!(a,i,v)
 Atomix.@atomic a[i]+=v
 nothing
end
include("../study_device.jl")
@kernel function scatter_entry!(parts,den,n)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_scatter!(i,parts,den,n)
end
@kernel function push_entry!(parts,ef,qm,n)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_push!(i,parts,ef,qm,n)
end
@kernel function rewind_entry!(parts,ef,qm,n)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_rewind!(i,parts,ef,qm,n)
end
function scatter_launch!(p,d,n)
 scatter_entry!(get_backend(p),256)(p,d,Int32(n);ndrange=(1+div(n,256))*256)
 nothing
end
function push_launch!(p,e,qm,n)
 push_entry!(get_backend(p),256)(p,e,qm,Int32(n);ndrange=(1+div(n,256))*256)
 nothing
end
function rewind_launch!(p,e,qm,n)
 rewind_entry!(get_backend(p),256)(p,e,qm,Int32(n);ndrange=(1+div(n,256))*256)
 nothing
end
include("../study_common.jl")
study()
