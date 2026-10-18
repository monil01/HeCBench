@inline function dense_body!(a,x,y,n,i)
 if i<=n
  value=0.0f0
  for j in 1:n
   v=@inbounds a[16+(i-1)*n+j]
   if v!=0.0f0
    value=muladd(v,@inbounds(x[16+j]),value)
   end
  end
  @inbounds y[16+i]=value
 end
 return nothing
end
