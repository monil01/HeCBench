using StaticArrays
@inline function tsa_pair(a,b,cellr,celli,kx,ky,physical,w,h,rl,im,::Val{BY},::Val{AXIS},::Val{BACK}) where {BY,AXIS,BACK}
 delta=Int32(1-2BACK)
 kpeerx=kx+(AXIS==0 ? delta : Int32(0));kpeery=ky+(AXIS==1 ? delta : Int32(0))
 limit=AXIS==0 ? w : h;localpos=AXIS==0 ? kx : ky;localmax=AXIS==0 ? Int32(16) : Int32(BY)
 if physical>=BACK && physical<limit-1+BACK && localpos>=BACK && localpos<localmax-1+BACK
  idx=kpeery*Int32(16)+kpeerx+Int32(1)
  @inbounds pr=rl[idx];pi=im[idx];rl[idx]=a*pr-b*celli;im[idx]=a*pi+b*cellr
  cellr,celli=a*cellr-b*pi,a*celli+b*pr
 end
 return cellr,celli
end
