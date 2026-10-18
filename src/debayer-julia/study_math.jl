@inline function reflect1(i,n)
 i=i<Int32(0) ? -i : i
 return i>=n ? Int32(2)*n-i-Int32(2) : i
end
@inline function load_apron!(tid,shared,data,w,h,bx,by)
 for f in tid:Int32(160):Int32(323)
  ar=div(f,Int32(36));ac=rem(f,Int32(36));gc=reflect1(ac+bx*Int32(32)-Int32(2),w);gr=reflect1(ar+by*Int32(5)-Int32(2),h)
  @inbounds shared[f+1]=Int32(data[gr*w+gc+1])
 end
 return
end
@inline F(a,c,r)=@inbounds a[r*Int32(36)+c+Int32(1)]
@inline sat8(x)=clamp(x,Int32(0),Int32(255))%UInt8
@inline function pixel!(tx,ty,bx,by,a,out,w,h,pat)
 i=tx+Int32(2);j=ty+Int32(2);gc=bx*Int32(32)+tx;gr=by*Int32(5)+ty;I=Int32
 fij=F(a,i,j)
 r1=div(I(4)*fij+I(2)*(F(a,i-I(1),j)+F(a,i,j-I(1))+F(a,i+I(1),j)+F(a,i,j+I(1)))-F(a,i-I(2),j)-F(a,i+I(2),j)-F(a,i,j-I(2))-F(a,i,j+I(2)),I(8))
 corners=F(a,i-I(1),j-I(1))+F(a,i+I(1),j-I(1))+F(a,i-I(1),j+I(1))+F(a,i+I(1),j+I(1))
 r2=div(I(8)*(F(a,i-I(1),j)+F(a,i+I(1),j))+I(10)*fij+F(a,i,j-I(2))+F(a,i,j+I(2))-I(2)*(corners+F(a,i-I(2),j)+F(a,i+I(2),j)),I(16))
 r3=div(I(8)*(F(a,i,j-I(1))+F(a,i,j+I(1)))+I(10)*fij+F(a,i-I(2),j)+F(a,i+I(2),j)-I(2)*(corners+F(a,i,j-I(2))+F(a,i,j+I(2))),I(16))
 r4=div(I(12)*fij-I(3)*(F(a,i-I(2),j)+F(a,i+I(2),j)+F(a,i,j-I(2))+F(a,i,j+I(2)))+I(4)*corners,I(16))
 redcol=(pat==I(1))|(pat==I(3));redrow=(pat==I(2))|(pat==I(3));inred=(gr&I(1))==I(redrow);isred=inred&&((gc&I(1))==I(redcol));isblue=(!inred)&&((gc&I(1))!=I(redcol));isgreen=!(isred||isblue)
 rr=sat8(fij*I(isred)+r4*I(isblue)+r2*I(isgreen&&inred)+r3*I(isgreen&&!inred));bb=sat8(fij*I(isblue)+r4*I(isred)+r3*I(isgreen&&inred)+r2*I(isgreen&&!inred));gg=sat8(fij*I(isgreen)+r1*I(!isgreen))
 if gc<w&&gr<h
  offset=gr*(I(4)*w)+gc*I(3)
  @inbounds begin out[offset+1]=rr;out[offset+2]=gg;out[offset+3]=bb;end
 end
 return
end
