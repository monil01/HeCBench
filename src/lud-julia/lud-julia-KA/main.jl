using CUDA,KernelAbstractions
@kernel function diagonal_kernel!(m,n,off)
 tx=(@index(Local,NTuple))[1]-1;tile=(@localmem Float32 (256,))
 for i in 0:15;@inbounds tile[i*16+tx+1]=m[(off+i)*n+off+tx+1];end
 @synchronize
 for i in 0:14
  if tx>i
   for j in 0:i-1;@inbounds tile[tx*16+i+1]-=tile[tx*16+j+1]*tile[j*16+i+1];end
   @inbounds tile[tx*16+i+1]/=tile[i*16+i+1]
  end
  @synchronize
  if tx>i
   for j in 0:i;@inbounds tile[(i+1)*16+tx+1]-=tile[(i+1)*16+j+1]*tile[j*16+tx+1];end
  end
  @synchronize
 end
 for i in 1:15;@inbounds m[(off+i)*n+off+tx+1]=tile[i*16+tx+1];end
end
@kernel function perimeter_kernel!(m,n,off)
 tx=(@index(Local,NTuple))[1]-1;bx=(@index(Group,NTuple))[1]-1
 dia=(@localmem Float32 (256,));pr=(@localmem Float32 (256,));pc=(@localmem Float32 (256,))
 if tx<16
  ix=tx
  for i in 0:7;@inbounds dia[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds pr[i*16+ix+1]=m[(off+i)*n+off+(bx+1)*16+ix+1];end
 else
  ix=tx-16
  for i in 8:15;@inbounds dia[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds pc[i*16+ix+1]=m[(off+(bx+1)*16+i)*n+off+ix+1];end
 end
 @synchronize
 if tx<16
  ix=tx
  for i in 1:15,j in 0:i-1;@inbounds pr[i*16+ix+1]-=dia[i*16+j+1]*pr[j*16+ix+1];end
 else
  ix=tx-16
  for i in 0:15
   for j in 0:i-1;@inbounds pc[ix*16+i+1]-=pc[ix*16+j+1]*dia[j*16+i+1];end
   @inbounds pc[ix*16+i+1]/=dia[i*16+i+1]
  end
 end
 @synchronize
 if tx<16
  ix=tx
  for i in 1:15;@inbounds m[(off+i)*n+off+(bx+1)*16+ix+1]=pr[i*16+ix+1];end
 else
  ix=tx-16
  for i in 0:15;@inbounds m[(off+(bx+1)*16+i)*n+off+ix+1]=pc[i*16+ix+1];end
 end
end
@kernel function internal_kernel!(m,n,off)
 tx=(@index(Local,NTuple))[1]-1;ty=(@index(Local,NTuple))[2]-1;bx=(@index(Group,NTuple))[1]-1;by=(@index(Group,NTuple))[2]-1
 pr=(@localmem Float32 (256,));pc=(@localmem Float32 (256,))
 row=off+(by+1)*16;col=off+(bx+1)*16
 @inbounds pr[ty*16+tx+1]=m[(off+ty)*n+col+tx+1]
 @inbounds pc[ty*16+tx+1]=m[(row+ty)*n+off+tx+1]
 @synchronize;sum=0f0
 for i in 0:15;@inbounds sum+=pc[ty*16+i+1]*pr[i*16+tx+1];end
 @inbounds m[(row+ty)*n+col+tx+1]-=sum
end

include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(m,n)
 backend=KernelAbstractions.get_backend(m)
 for off in 0:16:n-32
  tail=(n-off)÷16-1
  diagonal_kernel!(backend,16)(m,n,off;ndrange=16)
  perimeter_kernel!(backend,32)(m,n,off;ndrange=tail*32)
  internal_kernel!(backend,(16,16))(m,n,off;ndrange=(tail*16,tail*16))
 end
 diagonal_kernel!(backend,16)(m,n,n-16;ndrange=16)
end
study_main(study_launch!)
