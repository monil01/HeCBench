@inline function exp_fill!(s,A,B,sa,sb,m,n,tx,ty,bx,by)
 i=tx+bx*Int32(128);j=ty+by*Int32(32)
 for k in Int32(0):Int32(3)
  x=tx+k*Int32(32)+Int32(1)
  @inbounds s[x]=A[i+k*Int32(32)+Int32(1)]
  @inbounds s[Int32(128)+x]=A[m+i+k*Int32(32)+Int32(1)]
  @inbounds s[Int32(320)+x]=sa[i+k*Int32(32)+Int32(1)]
  if tx==0
   y=ty+k*Int32(8)+Int32(1)
   @inbounds s[Int32(256)+y]=B[j+k*Int32(8)+Int32(1)]
   @inbounds s[Int32(288)+y]=B[n+j+k*Int32(8)+Int32(1)]
   @inbounds s[Int32(448)+y]=sb[j+k*Int32(8)+Int32(1)]
  end
 end
 return
end
@inline function exp_sum(s,tx,ty,::Type{T}) where T
 acc=zero(T)
 for ti in Int32(0):Int32(3),tj in Int32(0):Int32(3)
  x=tx+ti*Int32(32)+Int32(1);y=ty+tj*Int32(8)+Int32(1)
  @inbounds dx=s[x]-s[Int32(256)+y]
  @inbounds dy=s[Int32(128)+x]-s[Int32(288)+y]
  dist=muladd(dy,dy,dx*dx)
  @inbounds acc+=exp(-dist/(s[Int32(320)+x]+s[Int32(448)+y]))
 end
 return acc
end
