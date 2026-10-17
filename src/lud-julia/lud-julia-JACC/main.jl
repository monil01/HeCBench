using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const TDIAG=CUDA.zeros(Float32,256)
const TPERI=CUDA.zeros(Float32,768)
const TINNER=CUDA.zeros(Float32,512)
@inline function diagonal_callback!(gi,m,n,off,template)
 tx=gi-1;shared=JACC.shared(template)
 for i in 0:15;@inbounds shared[i*16+tx+1]=m[(off+i)*n+off+tx+1];end
 JACC.sync_workgroup()
 for i in 0:14
  if tx>i
   for j in 0:i-1;@inbounds shared[tx*16+i+1]-=shared[tx*16+j+1]*shared[j*16+i+1];end
   @inbounds shared[tx*16+i+1]/=shared[i*16+i+1]
  end
  JACC.sync_workgroup()
  if tx>i
   for j in 0:i;@inbounds shared[(i+1)*16+tx+1]-=shared[(i+1)*16+j+1]*shared[j*16+tx+1];end
  end
  JACC.sync_workgroup()
 end
 for i in 1:15;@inbounds m[(off+i)*n+off+tx+1]=shared[i*16+tx+1];end
 return
end
@inline function perimeter_callback!(gi,m,n,off,template)
 tx=(gi-1)%32;bx=(gi-1)÷32
 shared=JACC.shared(template)
 if tx<16
  ix=tx
  for i in 0:7;@inbounds shared[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds shared[256+i*16+ix+1]=m[(off+i)*n+off+(bx+1)*16+ix+1];end
 else
  ix=tx-16
  for i in 8:15;@inbounds shared[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds shared[512+i*16+ix+1]=m[(off+(bx+1)*16+i)*n+off+ix+1];end
 end
 JACC.sync_workgroup()
 if tx<16
  ix=tx
  for i in 1:15,j in 0:i-1;@inbounds shared[256+i*16+ix+1]-=shared[i*16+j+1]*shared[256+j*16+ix+1];end
 else
  ix=tx-16
  for i in 0:15
   for j in 0:i-1;@inbounds shared[512+ix*16+i+1]-=shared[512+ix*16+j+1]*shared[j*16+i+1];end
   @inbounds shared[512+ix*16+i+1]/=shared[i*16+i+1]
  end
 end
 JACC.sync_workgroup()
 if tx<16
  ix=tx
  for i in 1:15;@inbounds m[(off+i)*n+off+(bx+1)*16+ix+1]=shared[256+i*16+ix+1];end
 else
  ix=tx-16
  for i in 0:15;@inbounds m[(off+(bx+1)*16+i)*n+off+ix+1]=shared[512+i*16+ix+1];end
 end
 return
end
@inline function internal_callback!(gi,gj,m,n,off,template)
 tx=(gi-1)%16;ty=(gj-1)%16;bx=(gi-1)÷16;by=(gj-1)÷16
 shared=JACC.shared(template)
 row=off+(by+1)*16;col=off+(bx+1)*16
 @inbounds shared[ty*16+tx+1]=m[(off+ty)*n+col+tx+1]
 @inbounds shared[256+ty*16+tx+1]=m[(row+ty)*n+off+tx+1]
 JACC.sync_workgroup();sum=0f0
 for i in 0:15;@inbounds sum+=shared[256+ty*16+i+1]*shared[i*16+tx+1];end
 @inbounds m[(row+ty)*n+col+tx+1]-=sum
 return
end


function study_launch!(m,n)
 backend=JACC.get_backend(:cuda)
 for off in 0:16:n-32
  tail=(n-off)÷16-1
  sd=JACC.LaunchSpec{typeof(backend)}(;threads=16,blocks=1,shmem_size=1024,sync=false)
  sp=JACC.LaunchSpec{typeof(backend)}(;threads=32,blocks=tail,shmem_size=3072,sync=false)
  si=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(tail,tail),shmem_size=2048,sync=false)
  JACC.parallel_for(diagonal_callback!,sd,16,m,n,off,TDIAG)
  JACC.parallel_for(perimeter_callback!,sp,tail*32,m,n,off,TPERI)
  JACC.parallel_for(internal_callback!,si,(tail*16,tail*16),m,n,off,TINNER)
 end
 sd=JACC.LaunchSpec{typeof(backend)}(;threads=16,blocks=1,shmem_size=1024,sync=false)
 JACC.parallel_for(diagonal_callback!,sd,16,m,n,n-16,TDIAG)
end
study_main(study_launch!)
