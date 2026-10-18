@inline function forward_value(x,i)
 v=x*1f0
 for it in Int32(0):Int32(255);v=v+(isodd(it+i) ? -1f0 : 1f0);end
 sin(v)
end
@inline function backward_value(a,g,i)
 for it in Int32(0):Int32(255);g=g+(isodd(it+i) ? a : -a);end
 sin(g)/cos(g)
end
@inline function prefetch_value(i,batch)
 v=Float32(i+batch)
 for it in Int32(0):Int32(255);v=v+(isodd(it+i) ? -1f0 : 1f0);end
 v
end
