using CUDA
struct Peak
 pos::UInt64
 val::Float32
end
function peak_cuda!(img,n,peaks)
 tid=Int32(threadIdx().x-1);bid=Int32(blockIdx().x-1);words=CUDA.CuStaticSharedArray(UInt32,768);val=0f0;pos=UInt64(0);i=UInt64(tid+256bid)
 while i<n
  v=@inbounds img[i+1]
  if abs(v)>abs(val);val=v;pos=i;end
  i+=UInt64(32768)
 end
 @inbounds begin;words[tid+1]=reinterpret(UInt32,val);words[257+2tid]=UInt32(pos&0xffffffff);words[258+2tid]=UInt32(pos>>32);end
 CUDA.sync_threads()
 if tid==0
  val=0f0;pos=UInt64(0)
  for j in Int32(0):Int32(255)
   v=@inbounds reinterpret(Float32,words[j+1])
   if abs(v)>abs(val);val=v;pos=UInt64(words[257+2j])|(UInt64(words[258+2j])<<32);end
  end
  @inbounds peaks[bid+1]=Peak(pos,val)
 end
 return
end
function subtract_cuda!(psf,res,w,sx,sy,ex,ey,dx,dy,val,gain)
 x=sx+Int32(threadIdx().x-1)+Int32(blockIdx().x-1)*Int32(16);y=sy+Int32(threadIdx().y-1)+Int32(blockIdx().y-1)*Int32(16)
 if x<=ex&&y<=ey
  @inbounds res[y*w+x+1]=muladd(-(gain*val),psf[(y-dy)*Int32(7)+x-dx+1],res[y*w+x+1])
 end
 return
end
