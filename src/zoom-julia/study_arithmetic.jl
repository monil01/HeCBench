@inline function load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch)
 rh=Float32(H)/Float32(Ho);rw=Float32(W)/Float32(Wo);bh=Int32(16)*by;bw=Int32(16)*bx
 sh=Int32(floor(Float32(bh)*rh));eh=Int32(ceil(Float32(bh+Int32(16))*rh));sw=Int32(floor(Float32(bw)*rw));ew=Int32(ceil(Float32(bw+Int32(16))*rw));stretch=ew-sw;im=Int64(bz)*pitch
 @inbounds for i in ty:Int32(16):eh-sh-Int32(1),j in tx:Int32(16):ew-sw-Int32(1)
  s[i*stretch+j+1]=(i+sh<H&&j+sw<W) ? input[im+(sh+i)*W+sw+j+1] : 0f0
 end
 return (rh,rw,bh,bw,sh,sw,stretch,im)
end
@inline function output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,::Val{MODE})where MODE
 rh,rw,bh,bw,sh,sw,stretch,im=state;oh=bh+ty;ow=bw+tx
 valid=oh<Ho&&ow<Wo
 if MODE==1;valid=valid&&oh>=hs&&oh<he&&ow>=ws&&ow<we;end
 if valid
  ah=Int32(floor(Float32(oh)*rh));zh=Int32(ceil(Float32(oh+Int32(1))*rh));aw=Int32(floor(Float32(ow)*rw));zw=Int32(ceil(Float32(ow+Int32(1))*rw));sum=0f0
  @inbounds for i in Int32(0):zh-ah-Int32(1),j in Int32(0):zw-aw-Int32(1);sum+=s[(ah+i-sh)*stretch+aw+j-sw+1];end
  dh=MODE==1 ? oh-hs : oh+hs;dw=MODE==1 ? ow-ws : ow+ws
  @inbounds out[im+dh*W+dw+1]=sum/Float32((zh-ah)*(zw-aw))
 end
 return
end
@inline function edge!(out,h,w,b,H,W,pitch,hs,ws,he,we)
 if h<H&&w<W&&(h<hs||h>=he||w<ws||w>=we)
  sh=clamp(h,hs,he-Int32(1));sw=clamp(w,ws,we-Int32(1));im=Int64(b)*pitch
  @inbounds out[im+h*W+w+1]=out[im+sh*W+sw+1]
 end
 return
end
