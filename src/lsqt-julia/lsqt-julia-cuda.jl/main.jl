# GPUQT Julia translation; originalGPLv3-or-later attribution in ../original_native/LICENCE.
using CUDA
const MODEL="cuda_jl"
const BACKEND=Val(:cuda_jl)
@inline block_sync(::Val{:cuda_jl})=CUDA.sync_threads()
@inline warp_sync(::Val{:cuda_jl})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
function kernel_zero!(n,re,im)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_zero!(i,n,re,im)
 nothing
end
function kernel_copy!(n,ar,ai,br,bi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_copy!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_add!(n,ar,ai,br,bi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_add!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_sz!(n,ar,ai,br,bi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_sz!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_ip1!(n,ar,ai,br,bi,orr,oi,offset)
 tid=Int32(threadIdx().x)-Int32(1);block=Int32(blockIdx().x)-Int32(1)
 sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)
 body_ip1!(tid,block,n,ar,ai,br,bi,orr,oi,offset,sr,si,BACKEND)
 nothing
end
function kernel_ip2!(n,ar,ai,orr,oi)
 tid=Int32(threadIdx().x)-Int32(1);block=Int32(blockIdx().x)-Int32(1)
 sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)
 body_ip2!(tid,block,n,ar,ai,orr,oi,sr,si,BACKEND)
 nothing
end
function kernel_H!(n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_H!(i,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 nothing
end
function kernel_commutator!(n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_commutator!(i,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function kernel_current!(n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_current!(i,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function kernel_cheb01!(n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_cheb01!(i,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 nothing
end
function kernel_cheb2!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_cheb2!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 nothing
end
function kernel_cheb1x!(n,r1x,i1x,orr,oi,b)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_cheb1x!(i,n,r1x,i1x,orr,oi,b)
 nothing
end
function kernel_cheb2x!(n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_cheb2x!(i,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 nothing
end
function kernel_poly!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)
 body_poly!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 nothing
end
function dispatch!(::Val{:zero},s,blocks,n,re,im)
 record_launch!(s,0)
 CUDA.@cuda threads=256 blocks=blocks kernel_zero!(n,re,im)
 nothing
end
function dispatch!(::Val{:copy},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,1)
 CUDA.@cuda threads=256 blocks=blocks kernel_copy!(n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:add},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,2)
 CUDA.@cuda threads=256 blocks=blocks kernel_add!(n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:sz},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,3)
 CUDA.@cuda threads=256 blocks=blocks kernel_sz!(n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:ip1},s,blocks,n,ar,ai,br,bi,orr,oi,offset)
 record_launch!(s,4)
 CUDA.@cuda threads=256 blocks=blocks kernel_ip1!(n,ar,ai,br,bi,orr,oi,offset)
 nothing
end
function dispatch!(::Val{:ip2},s,blocks,n,ar,ai,orr,oi)
 record_launch!(s,5)
 CUDA.@cuda threads=256 blocks=blocks kernel_ip2!(n,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:H},s,blocks,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 record_launch!(s,6)
 CUDA.@cuda threads=256 blocks=blocks kernel_H!(n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:commutator},s,blocks,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,7)
 CUDA.@cuda threads=256 blocks=blocks kernel_commutator!(n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:current},s,blocks,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,8)
 CUDA.@cuda threads=256 blocks=blocks kernel_current!(n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:cheb01},s,blocks,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 record_launch!(s,9)
 record_scalar!(s,9,(b0,b1,Float64(direction)))
 CUDA.@cuda threads=256 blocks=blocks kernel_cheb01!(n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 nothing
end
function dispatch!(::Val{:cheb2},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 record_launch!(s,10)
 record_scalar!(s,10,(b,Float64(label)))
 CUDA.@cuda threads=256 blocks=blocks kernel_cheb2!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 nothing
end
function dispatch!(::Val{:cheb1x},s,blocks,n,r1x,i1x,orr,oi,b)
 record_launch!(s,11)
 record_scalar!(s,11,(b,))
 CUDA.@cuda threads=256 blocks=blocks kernel_cheb1x!(n,r1x,i1x,orr,oi,b)
 nothing
end
function dispatch!(::Val{:cheb2x},s,blocks,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 record_launch!(s,12)
 record_scalar!(s,12,(b,Float64(label)))
 CUDA.@cuda threads=256 blocks=blocks kernel_cheb2x!(n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 nothing
end
function dispatch!(::Val{:poly},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 record_launch!(s,13)
 CUDA.@cuda threads=256 blocks=blocks kernel_poly!(n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 nothing
end
include("../study_common.jl")
main()
