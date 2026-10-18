using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function demosaic_kernel!(data,out,w,h,pat)
 tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1);bx=Int32(blockIdx().x-1);by=Int32(blockIdx().y-1);a=CuStaticSharedArray(Int32,(324,))
 load_apron!(ty*Int32(32)+tx,a,data,w,h,bx,by)
 sync_threads()
 pixel!(tx,ty,bx,by,a,out,w,h,pat)
 return
end
function launch(data,out,w,h,pat)
 @cuda threads=(32,5) blocks=(cld(Int(w),32),cld(Int(h),5)) demosaic_kernel!(data,out,w,h,pat)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
