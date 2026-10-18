using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function demosaic_kernel!(data,out,w,h,pat)
 tx=Int32(@index(Local,NTuple)[1]-1);ty=Int32(@index(Local,NTuple)[2]-1);bx=Int32(@index(Group,NTuple)[1]-1);by=Int32(@index(Group,NTuple)[2]-1);a=@localmem(Int32,(324,))
 load_apron!(ty*Int32(32)+tx,a,data,w,h,bx,by)
 @synchronize
 pixel!(tx,ty,bx,by,a,out,w,h,pat)
end
function launch(data,out,w,h,pat)
 demosaic_kernel!(BACKEND,(32,5))(data,out,w,h,pat;ndrange=(cld(Int(w),32)*32,cld(Int(h),5)*5))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
