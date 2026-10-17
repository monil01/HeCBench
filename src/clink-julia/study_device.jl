using StaticArrays
@inline function lstm_lane!(x,w,y,gid)
 h=MVector{5,Float32}(0,0,0,0,0);c=MVector{5,Float32}(0,0,0,0,0)
 ig=MVector{5,Float32}(0,0,0,0,0);fg=MVector{5,Float32}(0,0,0,0,0);og=MVector{5,Float32}(0,0,0,0,0);gg=MVector{5,Float32}(0,0,0,0,0)
 @inbounds for t in 1:257
  v=x[(gid-1)*257+t]
  for gate in 0:3
   for j in 1:5
    a=w[gate*5+j]*v
    for i in 1:5
     a=muladd(h[i],w[20+gate*25+(j-1)*5+i],a)
    end
    a+=w[120+gate*5+j]
    z=gate==3 ? tanh(a) : 1f0/(1f0+exp(-a))
    if gate==0;ig[j]=z;elseif gate==1;fg[j]=z;elseif gate==2;og[j]=z;else;gg[j]=z;end
   end
  end
  for j in 1:5
   c[j]=muladd(c[j],fg[j],gg[j]*ig[j]);h[j]=tanh(c[j])*og[j]
  end
  out=w[146]
  for j in 1:5;out=muladd(h[j],w[140+j],out);end
  y[(gid-1)*257+t]=out
 end
 return
end
