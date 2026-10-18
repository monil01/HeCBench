# Corrected-reference contract: true ratios, floating bilinear weights/pixels,
# original positive-away rounding and explicit half-center coordinate convention.
@inline function resize_site!(::Val{BILINEAR},out,inp,iter,stride,outsize,oh,ow,ih,iw,fy,fx) where BILINEAR
 inimage=ih*iw;outimage=oh*ow;required=Int32(outsize÷UInt64(outimage)÷UInt64(8))*outimage
 iter=Int32(iter);stride=Int32(stride)
 while iter<required
  cstart=(iter÷outimage)*Int32(8);y=(iter%outimage)÷ow;x=iter%ow
  if BILINEAR
   xx=max((Float32(x)+.5f0)*fx-.5f0,0f0);yy=max((Float32(y)+.5f0)*fy-.5f0,0f0)
   x0=trunc(Int32,xx);x1=min(x0+Int32(1),iw-Int32(1));y0=trunc(Int32,yy);y1=min(y0,ih-Int32(1));y2=min(y0+Int32(1),ih-Int32(1));r0=cstart*inimage+y1*iw;r1=cstart*inimage+y2*iw;oi=cstart*outimage+y*ow+x;u=xx-Float32(x0);v=yy-Float32(y0)
   c=cstart
   while c<cstart+Int32(8)
    @inbounds a=Float32(inp[r0+x0+Int32(1)]);@inbounds b=Float32(inp[r0+x1+Int32(1)]);@inbounds d=Float32(inp[r1+x0+Int32(1)]);@inbounds e=Float32(inp[r1+x1+Int32(1)])
    value=((a+v*(d-a))+u*(b-a))+(v*u)*(((e-b)-d)+a)
    @inbounds out[oi+Int32(1)]=unsafe_trunc(eltype(out),value)
    r0+=inimage;r1+=inimage;oi+=outimage;c+=Int32(1)
   end
  else
   # Nonnegative named coordinates make floor(z+.5) exactly lroundf-away.
   yy=min(trunc(Int32,(Float32(y)+.5f0)*fy+.5f0),ih-Int32(1));xx=min(trunc(Int32,(Float32(x)+.5f0)*fx+.5f0),iw-Int32(1));ii=cstart*inimage+yy*iw+xx;oi=cstart*outimage+y*ow+x
   for k in 1:8
    @inbounds out[oi+Int32(1)]=inp[ii+Int32(1)]
    ii+=inimage;oi+=outimage
   end
  end
  iter+=stride
 end
 nothing
end
