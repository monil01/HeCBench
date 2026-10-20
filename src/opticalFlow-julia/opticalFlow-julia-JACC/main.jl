using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
# A lazy halo descriptor, not a materialized array or CUDA device extension.
# JACC.shared's matrix method uses all32x6threads to fill one34x16 shared tile.
struct OpticalTileMatrix{A} <: AbstractMatrix{Float32}
 du::A;dv::A;w::Int32;h::Int32;s::Int32;bx::Int32;by::Int32
end
Base.size(::OpticalTileMatrix)=(34,16)
Base.IndexStyle(::Type{<:OpticalTileMatrix})=Base.IndexCartesian()
@inline function Base.getindex(t::OpticalTileMatrix,i::Integer,j::Integer)
 col=Int32(i)-Int32(2);r=Int32(j)-Int32(1);second=r>=Int32(8);row=(second ? r-Int32(8) : r)-Int32(1)
 x=clamp(t.bx*Int32(32)+col,Int32(0),t.w-Int32(1));y=clamp(t.by*Int32(6)+row,Int32(0),t.h-Int32(1));index=x+y*t.s+Int32(1)
 @inbounds return second ? t.dv[index] : t.du[index]
end
struct SharedPlane{A} <: AbstractVector{Float32}
 data::A;offset::Int32
end
Base.size(::SharedPlane)=(272,)
Base.IndexStyle(::Type{<:SharedPlane})=Base.IndexLinear()
@inline Base.getindex(t::SharedPlane,i::Integer)=@inbounds t.data[Int32(i)+t.offset]
function kernel_down!(i,j,src,sw,sh,ss,w,h,s,out);body_down!(Int32(i)-Int32(1),Int32(j)-Int32(1),src,sw,sh,ss,w,h,s,out);nothing;end
function kernel_up!(i,j,src,sw,sh,ss,w,h,s,scale,out);body_up!(Int32(i)-Int32(1),Int32(j)-Int32(1),src,sw,sh,ss,w,h,s,scale,out);nothing;end
function kernel_warp!(i,j,src,w,h,s,u,v,out);body_warp!(Int32(i)-Int32(1),Int32(j)-Int32(1),src,w,h,s,u,v,out);nothing;end
function kernel_derivatives!(i,j,a,b,w,h,s,gx,gy,gt);body_derivatives!(Int32(i)-Int32(1),Int32(j)-Int32(1),a,b,w,h,s,gx,gy,gt);nothing;end
function kernel_add!(index,a,b,n,out);body_add!(Int32(index)-Int32(1),a,b,n,out);nothing;end
function kernel_jacobi!(i,j,du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 ix=Int32(i)-Int32(1);iy=Int32(j)-Int32(1);bx=div(ix,Int32(32));by=div(iy,Int32(6));tx=rem(ix,Int32(32));ty=rem(iy,Int32(6))
 tile=JACC.shared(OpticalTileMatrix(du0,dv0,w,h,s,bx,by))
 # JACC.shared includes the workgroup barrier; no second barrier or scratch copy.
 jacobi_store!(tx,ty,bx,by,gx,gy,gt,w,h,s,alpha,du1,dv1,SharedPlane(tile,Int32(0)),SharedPlane(tile,Int32(272)))
 nothing
end
function dispatch!(::Val{:down},src,sw,sh,ss,w,h,s,out)
 sizes=(cld(w,32)*32,cld(h,8)*8);spec=JACC.launch_spec(threads=(32,8),blocks=(cld(w,32),cld(h,8)),shmem_size=0,sync=false)
 JACC.parallel_for(spec,sizes,kernel_down!,src,sw,sh,ss,w,h,s,out);CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:up},src,sw,sh,ss,w,h,s,scale,out)
 sizes=(cld(w,32)*32,cld(h,8)*8);spec=JACC.launch_spec(threads=(32,8),blocks=(cld(w,32),cld(h,8)),shmem_size=0,sync=false)
 JACC.parallel_for(spec,sizes,kernel_up!,src,sw,sh,ss,w,h,s,scale,out);CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:warp},src,w,h,s,u,v,out)
 sizes=(cld(w,32)*32,cld(h,6)*6);spec=JACC.launch_spec(threads=(32,6),blocks=(cld(w,32),cld(h,6)),shmem_size=0,sync=false)
 JACC.parallel_for(spec,sizes,kernel_warp!,src,w,h,s,u,v,out);CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:derivatives},a,b,w,h,s,gx,gy,gt)
 sizes=(cld(w,32)*32,cld(h,6)*6);spec=JACC.launch_spec(threads=(32,6),blocks=(cld(w,32),cld(h,6)),shmem_size=0,sync=false)
 JACC.parallel_for(spec,sizes,kernel_derivatives!,a,b,w,h,s,gx,gy,gt);CUDA.device_synchronize();nothing
end
function dispatch!(::Val{:jacobi},du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1)
 sizes=(cld(w,32)*32,cld(h,6)*6);spec=JACC.launch_spec(threads=(32,6),blocks=(cld(w,32),cld(h,6)),shmem_size=2176,sync=false)
 JACC.parallel_for(spec,sizes,kernel_jacobi!,du0,dv0,gx,gy,gt,w,h,s,alpha,du1,dv1);nothing
end
function dispatch!(::Val{:add},a,b,n,out)
 spec=JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false);JACC.parallel_for(spec,cld(n,256)*256,kernel_add!,a,b,n,out);nothing
end
include("../study_common.jl")
main()
