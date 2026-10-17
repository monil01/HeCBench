using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function sosfilt_jacc!(i,sos,zi,x,nsamples,template)
 T=eltype(x);tx=(i-1)%64;ty=(i-1)÷64;s=JACC.shared(template)

 @inbounds begin
 s[tx+1]=zero(T)
 for i=0:1;s[64+2*tx+i+1]=zi[ty*128+2*tx+i+1];end
 for i=0:5;s[192+6*tx+i+1]=sos[6*tx+i+1];end
 end
 JACC.sync_workgroup()
 temp=zero(T);xn=zero(T)
 for sample=0:62
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  JACC.sync_workgroup()
  @inbounds s[tx+1]=temp
  JACC.sync_workgroup()
 end
 for sample=63:nsamples-1
  @inbounds xn=tx==0 ? x[ty*nsamples+sample+1] : s[tx]
  @inbounds begin
 temp=s[192+6*tx+1]*xn+s[64+2*tx+1]
 s[64+2*tx+1]=s[192+6*tx+2]*xn-s[192+6*tx+5]*temp+s[64+2*tx+2]
 s[64+2*tx+2]=s[192+6*tx+3]*xn-s[192+6*tx+6]*temp
 end
  JACC.sync_workgroup()
  @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample-63+1]=temp;end
  JACC.sync_workgroup()
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
  JACC.sync_workgroup()
  if tx>sample
   @inbounds if tx<63;s[tx+1]=temp;else;x[ty*nsamples+sample+nsamples-63+1]=temp;end
  end
  JACC.sync_workgroup()
 end

 nothing
end
const SPEC_32_2=JACC.launch_spec(threads=64,blocks=2,shmem_size=2816,sync=false)
const SPEC_32_3=JACC.launch_spec(threads=64,blocks=3,shmem_size=3328,sync=false)
const SPEC_64_2=JACC.launch_spec(threads=64,blocks=2,shmem_size=5632,sync=false)
const SPEC_64_3=JACC.launch_spec(threads=64,blocks=3,shmem_size=6656,sync=false)
function launch!(signals,samples,::Val{SHARED},sos,zi,x,template) where SHARED
 spec=eltype(x)===Float32 ? (signals==2 ? SPEC_32_2 : SPEC_32_3) : (signals==2 ? SPEC_64_2 : SPEC_64_3)
 JACC.parallel_for(spec,signals*64,sosfilt_jacc!,sos,zi,x,samples,template)
 nothing
end
sos_study(launch!)
