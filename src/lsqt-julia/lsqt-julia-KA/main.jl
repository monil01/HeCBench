# GPUQT Julia translation, GPL-3.0-or-later; original notices in ../original_native/.
using CUDA, KernelAbstractions
const MODEL="ka"
const BACKEND=Val(:ka)
# Explicit CUDA-backend dependence in source-order shared/warp reductions.
@inline block_sync(::Val{:ka})=CUDA.sync_threads()
@inline warp_sync(::Val{:ka})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
@kernel function kernel_zero!(n,re,im)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_zero!(i,n,re,im)
 nothing
end
@kernel function kernel_copy!(n,ar,ai,br,bi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_copy!(i,n,ar,ai,br,bi)
 nothing
end
@kernel function kernel_add!(n,ar,ai,br,bi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_add!(i,n,ar,ai,br,bi)
 nothing
end
@kernel function kernel_sz!(n,ar,ai,br,bi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_sz!(i,n,ar,ai,br,bi)
 nothing
end
@kernel function kernel_ip1!(n,ar,ai,br,bi,orr,oi,offset)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)
 sr=@localmem Float64 (256,)
 si=@localmem Float64 (256,)
 body_ip1!(tid,block,n,ar,ai,br,bi,orr,oi,offset,sr,si,BACKEND)
 nothing
end
@kernel function kernel_ip2!(n,ar,ai,orr,oi)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)
 sr=@localmem Float64 (256,)
 si=@localmem Float64 (256,)
 body_ip2!(tid,block,n,ar,ai,orr,oi,sr,si,BACKEND)
 nothing
end
@kernel function kernel_H!(n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_H!(i,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 nothing
end
@kernel function kernel_commutator!(n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_commutator!(i,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
@kernel function kernel_current!(n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_current!(i,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
@kernel function kernel_cheb01!(n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_cheb01!(i,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 nothing
end
@kernel function kernel_cheb2!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_cheb2!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 nothing
end
@kernel function kernel_cheb1x!(n,r1x,i1x,orr,oi,b)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_cheb1x!(i,n,r1x,i1x,orr,oi,b)
 nothing
end
@kernel function kernel_cheb2x!(n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_cheb2x!(i,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 nothing
end
@kernel function kernel_poly!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 i=Int32(@index(Global,Linear))-Int32(1)
 body_poly!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 nothing
end
function dispatch!(::Val{:zero},s,blocks,n,re,im)
 record_launch!(s,0)
 kernel_zero!(CUDABackend(),256)(n,re,im;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:copy},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,1)
 kernel_copy!(CUDABackend(),256)(n,ar,ai,br,bi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:add},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,2)
 kernel_add!(CUDABackend(),256)(n,ar,ai,br,bi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:sz},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,3)
 kernel_sz!(CUDABackend(),256)(n,ar,ai,br,bi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:ip1},s,blocks,n,ar,ai,br,bi,orr,oi,offset)
 record_launch!(s,4)
 kernel_ip1!(CUDABackend(),256)(n,ar,ai,br,bi,orr,oi,offset;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:ip2},s,blocks,n,ar,ai,orr,oi)
 record_launch!(s,5)
 kernel_ip2!(CUDABackend(),256)(n,ar,ai,orr,oi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:H},s,blocks,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 record_launch!(s,6)
 kernel_H!(CUDABackend(),256)(n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:commutator},s,blocks,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,7)
 kernel_commutator!(CUDABackend(),256)(n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:current},s,blocks,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,8)
 kernel_current!(CUDABackend(),256)(n,nn,nl,hr,hi,dx,ar,ai,orr,oi;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:cheb01},s,blocks,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 record_launch!(s,9)
 record_scalar!(s,9,(b0,b1,Float64(direction)))
 kernel_cheb01!(CUDABackend(),256)(n,r0,i0,r1,i1,orr,oi,b0,b1,direction;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:cheb2},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 record_launch!(s,10)
 record_scalar!(s,10,(b,Float64(label)))
 kernel_cheb2!(CUDABackend(),256)(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:cheb1x},s,blocks,n,r1x,i1x,orr,oi,b)
 record_launch!(s,11)
 record_scalar!(s,11,(b,))
 kernel_cheb1x!(CUDABackend(),256)(n,r1x,i1x,orr,oi,b;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:cheb2x},s,blocks,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 record_launch!(s,12)
 record_scalar!(s,12,(b,Float64(label)))
 kernel_cheb2x!(CUDABackend(),256)(n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label;ndrange=blocks*256)
 nothing
end
function dispatch!(::Val{:poly},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 record_launch!(s,13)
 kernel_poly!(CUDABackend(),256)(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2;ndrange=blocks*256)
 nothing
end
include("../study_common.jl")
main()
