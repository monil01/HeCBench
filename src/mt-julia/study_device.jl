# Source MT private19wordstate and Cartesian Box–Muller, exact original recurrence.
using StaticArrays
@inline function body_mt!(gid,parameters,output,n)
 @inbounds begin
  matrix_a=parameters[gid*Int32(4)+Int32(1)];mask_b=parameters[gid*Int32(4)+Int32(2)];mask_c=parameters[gid*Int32(4)+Int32(3)]
  state=MVector{19,UInt32}(undef);state[1]=parameters[gid*Int32(4)+Int32(4)]
  for k=Int32(1):Int32(18)
   previous=state[k];state[k+Int32(1)]=UInt32(1812433253)*(previous⊻(previous>>30))+UInt32(k)
  end
  index=Int32(0);nextvalue=state[1]
  for j=Int32(0):n-Int32(1)
   next=index+Int32(1);middle=index+Int32(9)
   if next>=Int32(19);next-=Int32(19);end
   if middle>=Int32(19);middle-=Int32(19);end
   current=nextvalue;nextvalue=state[next+Int32(1)];mid=state[middle+Int32(1)]
   x=(current&UInt32(0xfffffffe))|(nextvalue&UInt32(1))
   x=mid⊻(x>>1)⊻((x&UInt32(1))!=UInt32(0) ? matrix_a : UInt32(0))
   state[index+Int32(1)]=x;index=next
   x=x⊻(x>>12);x=x⊻((x<<7)&mask_b);x=x⊻((x<<15)&mask_c);x=x⊻(x>>18)
   output[gid+j*Int32(4096)+Int32(1)]=(Float32(x)+1f0)/4294967296f0
  end
 end
 nothing
end
@inline function body_boxmuller!(gid,output,n)
 @inbounds for j=Int32(0):Int32(2):n-Int32(1)
  i=gid+j*Int32(4096)+Int32(1);u1=output[i];u2=output[i+Int32(4096)]
  r=sqrt(-2f0*log(u1));phi=(2f0*3.14159265358979f0)*u2
  output[i]=r*cos(phi);output[i+Int32(4096)]=r*sin(phi)
 end
 nothing
end
