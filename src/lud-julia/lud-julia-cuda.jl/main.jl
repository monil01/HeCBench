using CUDA
function diagonal_kernel!(m,n,off)
 tx=Int(threadIdx().x)-1;tile=CuStaticSharedArray(Float32,256)
 for i in 0:15;@inbounds tile[i*16+tx+1]=m[(off+i)*n+off+tx+1];end
 sync_threads()
 for i in 0:14
  if tx>i
   for j in 0:i-1;@inbounds tile[tx*16+i+1]-=tile[tx*16+j+1]*tile[j*16+i+1];end
   @inbounds tile[tx*16+i+1]/=tile[i*16+i+1]
  end
  sync_threads()
  if tx>i
   for j in 0:i;@inbounds tile[(i+1)*16+tx+1]-=tile[(i+1)*16+j+1]*tile[j*16+tx+1];end
  end
  sync_threads()
 end
 for i in 1:15;@inbounds m[(off+i)*n+off+tx+1]=tile[i*16+tx+1];end
 return
end
function perimeter_kernel!(m,n,off)
 tx=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1
 dia=CuStaticSharedArray(Float32,256);pr=CuStaticSharedArray(Float32,256);pc=CuStaticSharedArray(Float32,256)
 if tx<16
  ix=tx
  for i in 0:7;@inbounds dia[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds pr[i*16+ix+1]=m[(off+i)*n+off+(bx+1)*16+ix+1];end
 else
  ix=tx-16
  for i in 8:15;@inbounds dia[i*16+ix+1]=m[(off+i)*n+off+ix+1];end
  for i in 0:15;@inbounds pc[i*16+ix+1]=m[(off+(bx+1)*16+i)*n+off+ix+1];end
 end
 sync_threads()
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
 sync_threads()
 if tx<16
  ix=tx
  for i in 1:15;@inbounds m[(off+i)*n+off+(bx+1)*16+ix+1]=pr[i*16+ix+1];end
 else
  ix=tx-16
  for i in 0:15;@inbounds m[(off+(bx+1)*16+i)*n+off+ix+1]=pc[i*16+ix+1];end
 end
 return
end
function internal_kernel!(m,n,off)
 tx=Int(threadIdx().x)-1;ty=Int(threadIdx().y)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1
 pr=CuStaticSharedArray(Float32,256);pc=CuStaticSharedArray(Float32,256)
 row=off+(by+1)*16;col=off+(bx+1)*16
 @inbounds pr[ty*16+tx+1]=m[(off+ty)*n+col+tx+1]
 @inbounds pc[ty*16+tx+1]=m[(row+ty)*n+off+tx+1]
 sync_threads();sum=0f0
 for i in 0:15;@inbounds sum+=pc[ty*16+i+1]*pr[i*16+tx+1];end
 @inbounds m[(row+ty)*n+col+tx+1]-=sum
 return
end
