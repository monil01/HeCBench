using CUDA
function sosfilt_cuda!(sos,zi,x,nsamples,::Val{SHARED}) where SHARED
 T=eltype(x);tx=Int(threadIdx().x)-1;ty=Int(blockIdx().x)-1;s=CuDynamicSharedArray(T,(SHARED,))

 @inbounds begin
 s[tx+1]=zero(T)
 for i=0:1;s[64+2*tx+i+1]=zi[ty*128+2*tx+i+1];end
 for i=0:5;s[192+6*tx+i+1]=sos[6*tx+i+1];end
 end
 CUDA.sync_threads()
 temp=zero(T);xn=zero(T)
 for sample=0:62
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  CUDA.sync_threads()
  @inbounds s[tx+1]=temp
  CUDA.sync_threads()
 end
 for sample=63:nsamples-1
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  CUDA.sync_threads()
  @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample-63+1]=temp;end
  CUDA.sync_threads()
 end
 for sample=0:63
  if tx>sample
   @inbounds xn=s[tx]
   @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  end
  CUDA.sync_threads()
  if tx>sample
   @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample+nsamples-63+1]=temp;end
  end
  CUDA.sync_threads()
 end

 return
end
