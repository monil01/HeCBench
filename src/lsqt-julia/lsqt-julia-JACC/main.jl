# GPUQT Julia translation, GPL-3.0-or-later; original notices in ../original_native/.
using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
const BACKEND=Val(:jacc)
@inline block_sync(::Val{:jacc})=JACC.sync_workgroup()
# Explicit CUDA-backend dependency for legacy source warp-order reductions.
@inline warp_sync(::Val{:jacc})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
function kernel_zero!(index,n,re,im)
 i=Int32(index)-Int32(1)
 body_zero!(i,n,re,im)
 nothing
end
function kernel_copy!(index,n,ar,ai,br,bi)
 i=Int32(index)-Int32(1)
 body_copy!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_add!(index,n,ar,ai,br,bi)
 i=Int32(index)-Int32(1)
 body_add!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_sz!(index,n,ar,ai,br,bi)
 i=Int32(index)-Int32(1)
 body_sz!(i,n,ar,ai,br,bi)
 nothing
end
function kernel_ip1!(index,n,ar,ai,br,bi,orr,oi,offset)
 i=Int32(index)-Int32(1);tid=rem(i,Int32(256));block=div(i,Int32(256))
 sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)
 body_ip1!(tid,block,n,ar,ai,br,bi,orr,oi,offset,sr,si,BACKEND)
 nothing
end
function kernel_ip2!(index,n,ar,ai,orr,oi)
 i=Int32(index)-Int32(1);tid=rem(i,Int32(256));block=div(i,Int32(256))
 sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)
 body_ip2!(tid,block,n,ar,ai,orr,oi,sr,si,BACKEND)
 nothing
end
function kernel_H!(index,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 i=Int32(index)-Int32(1)
 body_H!(i,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 nothing
end
function kernel_commutator!(index,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=Int32(index)-Int32(1)
 body_commutator!(i,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function kernel_current!(index,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 i=Int32(index)-Int32(1)
 body_current!(i,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function kernel_cheb01!(index,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 i=Int32(index)-Int32(1)
 body_cheb01!(i,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 nothing
end
function kernel_cheb2!(index,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 i=Int32(index)-Int32(1)
 body_cheb2!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 nothing
end
function kernel_cheb1x!(index,n,r1x,i1x,orr,oi,b)
 i=Int32(index)-Int32(1)
 body_cheb1x!(i,n,r1x,i1x,orr,oi,b)
 nothing
end
function kernel_cheb2x!(index,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 i=Int32(index)-Int32(1)
 body_cheb2x!(i,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 nothing
end
function kernel_poly!(index,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 i=Int32(index)-Int32(1)
 body_poly!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 nothing
end
function dispatch!(::Val{:zero},s,blocks,n,re,im)
 record_launch!(s,0)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_zero!,n,re,im)
 nothing
end
function dispatch!(::Val{:copy},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,1)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_copy!,n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:add},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,2)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_add!,n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:sz},s,blocks,n,ar,ai,br,bi)
 record_launch!(s,3)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_sz!,n,ar,ai,br,bi)
 nothing
end
function dispatch!(::Val{:ip1},s,blocks,n,ar,ai,br,bi,orr,oi,offset)
 record_launch!(s,4)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_ip1!,n,ar,ai,br,bi,orr,oi,offset)
 nothing
end
function dispatch!(::Val{:ip2},s,blocks,n,ar,ai,orr,oi)
 record_launch!(s,5)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_ip2!,n,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:H},s,blocks,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 record_launch!(s,6)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_H!,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:commutator},s,blocks,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,7)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_commutator!,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:current},s,blocks,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 record_launch!(s,8)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_current!,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 nothing
end
function dispatch!(::Val{:cheb01},s,blocks,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 record_launch!(s,9)
 record_scalar!(s,9,(b0,b1,Float64(direction)))
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_cheb01!,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 nothing
end
function dispatch!(::Val{:cheb2},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 record_launch!(s,10)
 record_scalar!(s,10,(b,Float64(label)))
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_cheb2!,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 nothing
end
function dispatch!(::Val{:cheb1x},s,blocks,n,r1x,i1x,orr,oi,b)
 record_launch!(s,11)
 record_scalar!(s,11,(b,))
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_cheb1x!,n,r1x,i1x,orr,oi,b)
 nothing
end
function dispatch!(::Val{:cheb2x},s,blocks,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 record_launch!(s,12)
 record_scalar!(s,12,(b,Float64(label)))
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_cheb2x!,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 nothing
end
function dispatch!(::Val{:poly},s,blocks,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 record_launch!(s,13)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,kernel_poly!,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 nothing
end
include("../study_common.jl")
main()
