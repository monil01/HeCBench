function fixture(c)
 a=Float32[];b=Float32[]
 for i in 0:127,d in 0:127
  push!(a,c==0 ? Float32(mod(i*13+d*7,17)-8)/16f0 : i==0 ? 0f0 : d==mod(i,128) ? 1f0 : 0f0)
  push!(b,c==0 ? Float32(mod(i*19+d*11,23)-11)/16f0 : i==0||i==127 ? 1f0 : -1f0)
 end
 return a,b
end
function oracle(a,b,k)
 scores=Float32[];indices=Int32[]
 order=k<=5 ? collect(0:127) : sort(collect(0:127);by=i->(div(mod(i,k==6 ? 16 : 32),4),div(i,k==6 ? 16 : 32),mod(i,4)))
 for i in 0:127
  best=0;idx=-1
  for j in order
   v=sum(Int64(round(a[i*128+d]*16))*Int64(round(b[j*128+d]*16)) for d in 1:128)
   if v>best;best=v;idx=j;end
  end
  push!(scores,Float32(best)/256f0);push!(indices,Int32(idx))
 end
 return scores,indices
end
