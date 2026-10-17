@inline function entropy_point!(out,input,table,counts,x,y,lane,w,h)
 @inbounds begin
  for k in 0:15;counts[k*256+lane]=Int32(0);end
  total=Int32(0)
  for dy in -2:2,dx in -2:2
   xx=x+dx;yy=y+dy
   if 1<=xx<=w && 1<=yy<=h
    k=Int(input[(yy-1)*w+xx]);counts[k*256+lane]+=Int32(1);total+=Int32(1)
   end
  end
  s=0.0f0
  for k in 0:15;s-=table[Int(counts[k*256+lane])+1];end
  out[(y-1)*w+x]=s/Float32(total)+log2(Float32(total))
 end
 nothing
end
