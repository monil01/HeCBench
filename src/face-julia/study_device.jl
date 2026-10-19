@inline function address_body!(rects,out,base,width,n,gid)
 if gid<n
  idx=gid*Int32(12)
  for k in Int32(0):Int32(2)
   j=idx+k*Int32(4);x=rects[j+1];y=rects[j+2];w=rects[j+3];h=rects[j+4]
   missing=k==2&&x==0&&y==0&&w==0&&h==0
   out[j+1]=missing ? UInt64(0) : base+UInt64(width*y+x)*UInt64(4)
   out[j+2]=missing ? UInt64(0) : base+UInt64(width*y+x+w)*UInt64(4)
   out[j+3]=missing ? UInt64(0) : base+UInt64(width*(y+h)+x)*UInt64(4)
   out[j+4]=missing ? UInt64(0) : base+UInt64(width*(y+h)+x+w)*UInt64(4)
  end
 end
 nothing
end
