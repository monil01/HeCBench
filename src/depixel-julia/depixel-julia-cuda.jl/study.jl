using CUDA
include(joinpath(@__DIR__,"..","depixel_math.jl"))
function dep_check_kernel!(a,t,w,h)
 i=Int32(blockIdx().x-1)*Int32(256)+Int32(threadIdx().x-1)
 dep_check!(a,t,i,w,h)
 return
end
function dep_eliminate_kernel!(t,o,w,h)
 i=Int32(blockIdx().x-1)*Int32(256)+Int32(threadIdx().x-1)
 dep_eliminate!(t,o,i,w,h)
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function dep_launch!(a,t,o,w,h,n)
 @cuda threads=256 blocks=n÷256 dep_check_kernel!(a,t,w,h)
 @cuda threads=256 blocks=n÷256 dep_eliminate_kernel!(t,o,w,h)
end
depixel_study(dep_launch!)
