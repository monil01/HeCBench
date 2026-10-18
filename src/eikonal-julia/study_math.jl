@inline function source_time(a,b,c,s)
 if a<b;a,b=b,a;end
 if b<c;b,c=c,b;end
 if a<b;a,b=b,a;end
 ret=1e20
 if c<1e20
  ret=c+s
  if ret>b
   tmp=((b+c)+Float64(sqrt(Float32(2.0*s*s-(b-c)*(b-c)))))*0.5
   if tmp>b;ret=tmp;end
   if ret>a
    tmp=(a+b+c)/3.0+Float64(sqrt(Float32(2.0*(a*(b-a)+b*(c-b)+c*(a-c))+3.0*s*s)))/3.0
    if tmp>a;ret=tmp;end
   end
  end
 end
 return ret
end
@inline smidx(x,y,z)=x*Int32(36)+y*Int32(6)+z+Int32(1)
@inline function globalidx(x,y,z,n)
 x=clamp(x,Int32(0),n-Int32(1));y=clamp(y,Int32(0),n-Int32(1));z=clamp(z,Int32(0),n-Int32(1));nx=n÷Int32(4)
 return((z÷Int32(4)*nx+y÷Int32(4))*nx+x÷Int32(4))*Int32(64)+(z%Int32(4))*Int32(16)+(y%Int32(4))*Int32(4)+x%Int32(4)+Int32(1)
end
