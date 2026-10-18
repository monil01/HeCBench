@inline function mandel_count(i,j,rows,cols,cap)
 cr=muladd(Float32(i),2f0/Float32(rows),-1.5f0);ci=muladd(Float32(j),2f0/Float32(cols),-1f0);x=0f0;y=0f0;count=Int32(0)
 while count<cap
  xx=x*x;yy=y*y;radius=xx+yy
  if radius>=4f0;break;end
  difference=xx-yy;nx=difference+cr;twice=x+x;y=muladd(twice,y,ci);x=nx;count+=Int32(1)
 end
 return count
end
