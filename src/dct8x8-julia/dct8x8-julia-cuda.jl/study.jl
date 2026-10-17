include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(y,x,s,h,w,dir)
 @cuda threads=(16,16) blocks=(cld(w,16),cld(h,16)) dct8x8_kernel!(y,x,Int32(s),Int32(h),Int32(w),Int32(dir))
end
study_main(study_launch!)
