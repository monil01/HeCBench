using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
@inline function density_add!(a,i,v)
 JACC.@atomic a[i]+=v
 nothing
end
include("../study_device.jl")
function scatter_entry!(index,parts,den,n)
 body_scatter!(Int32(index)-Int32(1),parts,den,n)
 nothing
end
function push_entry!(index,parts,ef,qm,n)
 body_push!(Int32(index)-Int32(1),parts,ef,qm,n)
 nothing
end
function rewind_entry!(index,parts,ef,qm,n)
 body_rewind!(Int32(index)-Int32(1),parts,ef,qm,n)
 nothing
end
function scatter_launch!(p,d,n)
 blocks=1+div(n,256);spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,scatter_entry!,p,d,Int32(n))
 nothing
end
function push_launch!(p,e,qm,n)
 blocks=1+div(n,256);spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,push_entry!,p,e,qm,Int32(n))
 nothing
end
function rewind_launch!(p,e,qm,n)
 blocks=1+div(n,256);spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,rewind_entry!,p,e,qm,Int32(n))
 nothing
end
include("../study_common.jl")
study()
