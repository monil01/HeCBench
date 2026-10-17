using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function sosfilt_ka!(sos::AbstractVector{T},zi,x,nsamples,::Val{SHARED}) where {T,SHARED}
 tx=@index(Local,Linear)-1;ty=@index(Group,Linear)-1;s=@localmem T (SHARED,)

 @inbounds begin
 s[tx+1]=zero(T)
 for i=0:1;s[64+2*tx+i+1]=zi[ty*128+2*tx+i+1];end
 for i=0:5;s[192+6*tx+i+1]=sos[6*tx+i+1];end
 end
 @synchronize
 temp=zero(T);xn=zero(T)
 for sample=0:62
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  @synchronize
  @inbounds s[tx+1]=temp
  @synchronize
 end
 for sample=63:nsamples-1
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  @synchronize
  @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample-63+1]=temp;end
  @synchronize
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
  @synchronize
  if tx>sample
   @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample+nsamples-63+1]=temp;end
  end
  @synchronize
 end

end
const BACKEND=CUDABackend()
function launch!(signals,samples,::Val{SHARED},sos,zi,x,template) where SHARED
 sosfilt_ka!(BACKEND,64)(sos,zi,x,samples,Val(SHARED);ndrange=signals*64)
 nothing
end
sos_study(launch!)
