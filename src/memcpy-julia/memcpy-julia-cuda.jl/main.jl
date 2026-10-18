using CUDA
unwrap_host(x)=x
unwrap_host(x::SubArray{T,1,<:Array}) where T=unsafe_wrap(Array,pointer(x),length(x);own=false)
transfer!(dest,source)=copyto!(unwrap_host(dest),unwrap_host(source))
include("../study_common.jl")
run_study(transfer!)
