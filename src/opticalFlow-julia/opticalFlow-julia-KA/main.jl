using CUDA,KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel function kernel_down!(src,sw,sh,ss,w,h,s,out)
 index=@index(Global,NTuple);body_down!(Int32(index[1])-Int32(1),Int32(index[2])-Int32(1),src,sw,sh,ss,w,h,s,out)
end
@kernel function kernel_up!(src,sw,sh,ss,w,h,s,scale,out)
 index=@index(Global,NTuple);body_up!(Int32(index[1])-Int32(1),Int32(index[2])-Int32(1),src,sw,sh,ss,w,h,s,scale,out)
end
@kernel function kernel_warp!(src,w,h,s,u,v,out)
 index=@index(Global,NTuple);body_warp!(Int32(index[1])-Int32(1),Int32(index[2])-Int32(1),src,w,h,s,u,v,out)
end
@kernel function kernel_derivatives!(a,b,w,h,s,gx,gy,gt)
 index=@index(Global,NTuple);body_derivatives!(Int32(index[1])-Int32(1),Int32(index[2])-Int32(1),a,b,w,h,s,gx,gy,gt)
end
@kernel function kernel_add!(a,b,n,out)
 index=@index(Global,Linear);body_add!(Int32(index)-Int32(1),a,b,n,out)
end
@kernel function kernel_jacobi!(du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 du=@localmem(Float32,(272,));dv=@localmem(Float32,(272,))
 localindex=@index(Local,NTuple);groupindex=@index(Group,NTuple)
 tx=Int32(localindex[1])-Int32(1);ty=Int32(localindex[2])-Int32(1);bx=Int32(groupindex[1])-Int32(1);by=Int32(groupindex[2])-Int32(1)
 jacobi_load!(tx,ty,bx,by,du0,dv0,w,h,s,du,dv)
 @synchronize
 jacobi_store!(tx,ty,bx,by,gx,gy,gt,w,h,s,alpha,du1,dv1,du,dv)
end
function dispatch!(::Val{:down},src,sw,sh,ss,w,h,s,out)
 kernel_down!(get_backend(out),(32,8))(src,sw,sh,ss,w,h,s,out;ndrange=(cld(w,32)*32,cld(h,8)*8));CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:up},src,sw,sh,ss,w,h,s,scale,out)
 kernel_up!(get_backend(out),(32,8))(src,sw,sh,ss,w,h,s,scale,out;ndrange=(cld(w,32)*32,cld(h,8)*8));CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:warp},src,w,h,s,u,v,out)
 kernel_warp!(get_backend(out),(32,6))(src,w,h,s,u,v,out;ndrange=(cld(w,32)*32,cld(h,6)*6));CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:derivatives},a,b,w,h,s,gx,gy,gt)
 kernel_derivatives!(get_backend(gx),(32,6))(a,b,w,h,s,gx,gy,gt;ndrange=(cld(w,32)*32,cld(h,6)*6));CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:jacobi},du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 kernel_jacobi!(get_backend(du1),(32,6))(du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1;ndrange=(cld(w,32)*32,cld(h,6)*6));nothing
end
function dispatch!(::Val{:add},a,b,n,out)
 kernel_add!(get_backend(out),256)(a,b,n,out;ndrange=cld(n,256)*256);nothing
end
include("../study_common.jl")
main()
