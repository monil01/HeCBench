using CUDA
import JACC
JACC.@init_backend
const TEMPLATE=JACC.zeros(Int32,324)
include(joinpath(@__DIR__,"..","study_math.jl"))
function demosaic_kernel!(xx,yy,data,out,w,h,pat,template)
 tx=Int32(mod(xx-1,32));ty=Int32(mod(yy-1,5));bx=Int32(div(xx-1,32));by=Int32(div(yy-1,5));a=JACC.shared(template)
 load_apron!(ty*Int32(32)+tx,a,data,w,h,bx,by)
 JACC.sync_workgroup()
 pixel!(tx,ty,bx,by,a,out,w,h,pat)
 return
end
function launch(data,out,w,h,pat)
 blocks=(cld(Int(w),32),cld(Int(h),5));spec=JACC.launch_spec(threads=(32,5),blocks=blocks,shmem_size=1296,sync=false)
 JACC.parallel_for(spec,(blocks[1]*32,blocks[2]*5),demosaic_kernel!,data,out,w,h,pat,TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
