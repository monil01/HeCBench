using CUDA, KernelAbstractions
struct Peak
 pos::UInt64
 val::Float32
end
@kernel function peak_ka!(img,n,peaks)
 tid=@index(Local,Linear);bid=@index(Group,Linear);words=@localmem UInt32 (768,);val=0f0;pos=UInt64(0);i=UInt64((tid-1)+256*(bid-1))
 while i<n
  v=@inbounds img[i+1]
  if abs(v)>abs(val);val=v;pos=i;end
  i+=UInt64(32768)
 end
 @inbounds begin;words[tid]=reinterpret(UInt32,val);words[257+2*(tid-1)]=UInt32(pos&0xffffffff);words[258+2*(tid-1)]=UInt32(pos>>32);end
 @synchronize
 if tid==1
  bestval=0f0;bestpos=UInt64(0)
  for j in 0:255
   v=@inbounds reinterpret(Float32,words[j+1])
   if abs(v)>abs(bestval);bestval=v;bestpos=UInt64(words[257+2j])|(UInt64(words[258+2j])<<32);end
  end
  @inbounds peaks[bid]=Peak(bestpos,bestval)
 end
end
@kernel function subtract_ka!(psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
 lx,ly=@index(Local,NTuple);bx,by=@index(Group,NTuple);x=sx+Int32(lx-1)+Int32(bx-1)*Int32(16);y=sy+Int32(ly-1)+Int32(by-1)*Int32(16)
 if x<=ex&&y<=ey
  @inbounds res[y*w+x+1]=muladd(-(gain*val),psf[(y-dy)*Int32(7)+x-dx+1],res[y*w+x+1])
 end
end
