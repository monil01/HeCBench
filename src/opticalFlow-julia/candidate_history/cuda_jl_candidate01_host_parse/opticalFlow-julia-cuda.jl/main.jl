using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
@inline xy()=((Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1),(Int32(blockIdx().y)-Int32(1))*Int32(blockDim().y)+Int32(threadIdx().y)-Int32(1))
function kernel_down!(src,sw,sh,ss,w,h,s,out);x,y=xy();body_down!(x,y,src,sw,sh,ss,w,h,s,out);nothing;end
function kernel_up!(src,sw,sh,ss,w,h,s,scale,out);x,y=xy();body_up!(x,y,src,sw,sh,ss,w,h,s,scale,out);nothing;end
function kernel_warp!(src,w,h,s,u,v,out);x,y=xy();body_warp!(x,y,src,w,h,s,u,v,out);nothing;end
function kernel_derivatives!(a,b,w,h,s,gx,gy,gt);x,y=xy();body_derivatives!(x,y,a,b,w,h,s,gx,gy,gt);nothing;end
function kernel_add!(a,b,n,out)
 index=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1);body_add!(index,a,b,n,out);nothing
end
function kernel_jacobi!(du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 du=CUDA.CuStaticSharedArray(Float32,272);dv=CUDA.CuStaticSharedArray(Float32,272)
 tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1)
 jacobi_load!(tx,ty,bx,by,du0,dv0,w,h,s,du,dv);CUDA.sync_threads();jacobi_store!(tx,ty,bx,by,gx,gy,gt,w,h,s,alpha,du1,dv1,du,dv)
 nothing
end
function dispatch!(::Val{:down},src,sw,sh,ss,w,h,s,out)
 CUDA.@cuda threads=(32,8) blocks=(cld(w,32),cld(h,8)) kernel_down!(src,sw,sh,ss,w,h,s,out)
 CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:up},src,sw,sh,ss,w,h,s,scale,out)
 CUDA.@cuda threads=(32,8) blocks=(cld(w,32),cld(h,8)) kernel_up!(src,sw,sh,ss,w,h,s,scale,out)
 CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:warp},src,w,h,s,u,v,out)
 CUDA.@cuda threads=(32,6) blocks=(cld(w,32),cld(h,6)) kernel_warp!(src,w,h,s,u,v,out)
 CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:derivatives},a,b,w,h,s,gx,gy,gt)
 CUDA.@cuda threads=(32,6) blocks=(cld(w,32),cld(h,6)) kernel_derivatives!(a,b,w,h,s,gx,gy,gt)
 CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:jacobi},du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 CUDA.@cuda threads=(32,6) blocks=(cld(w,32),cld(h,6)) kernel_jacobi!(du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 nothing
end
function dispatch!(::Val{:add},a,b,n,out)
 CUDA.@cuda threads=256 blocks=cld(n,256) kernel_add!(a,b,n,out)
 nothing
end
include("../study_common.jl")
main()
