using CUDA
const MODEL="cuda_jl"
@inline density_add!(a,i,v)=CUDA.atomic_add!(pointer(a,i),v)
include("../study_device.jl")
function scatter_entry!(parts,den,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 body_scatter!(i,parts,den,n)
 nothing
end
function push_entry!(parts,ef,qm,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 body_push!(i,parts,ef,qm,n)
 nothing
end
function rewind_entry!(parts,ef,qm,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 body_rewind!(i,parts,ef,qm,n)
 nothing
end
function scatter_launch!(p,d,n)
 CUDA.@cuda threads=256 blocks=1+div(n,256) scatter_entry!(p,d,Int32(n))
 nothing
end
function push_launch!(p,e,qm,n)
 CUDA.@cuda threads=256 blocks=1+div(n,256) push_entry!(p,e,qm,Int32(n))
 nothing
end
function rewind_launch!(p,e,qm,n)
 CUDA.@cuda threads=256 blocks=1+div(n,256) rewind_entry!(p,e,qm,Int32(n))
 nothing
end
include("../study_common.jl")
study()
