const C0=Int32(0);const C1=Int32(1);const L200=Int32(200)
@inline function correlation!(x,y,mask,offset,wid,k,i,D,gene)
 sx=0f0;sxx=0f0;sy=0f0;sxy=0f0;syy=0f0;sn=0f0;sxxn=0f0;syn=0f0;sxyn=0f0;syyn=0f0
 @inbounds for j in C0:Int32(D-C1)
  a=gene[k*(D+C1)+j+C1];x[j+C1]=a
  if mask[offset+j+C1]==UInt8(49);sx+=a;else;sn+=a;end
 end
 sx/=Float32(wid);sn/=Float32(D-wid)
 @inbounds for j in C0:Int32(D-C1)
  if mask[offset+j+C1]==UInt8(49);v=sx-x[j+C1];sxx+=v*v
  else;v=sn-x[j+C1];sxxn+=v*v;end
 end
 sxx=sqrt(sxx);sxxn=sqrt(sxxn)
 @inbounds for j in C0:Int32(D-C1)
  a=gene[i*(D+C1)+j+C1];y[j+C1]=a
  if mask[offset+j+C1]==UInt8(49);sy+=a;else;syn+=a;end
 end
 sy/=Float32(wid);syn/=Float32(D-wid)
 @inbounds for j in C0:Int32(D-C1)
  if mask[offset+j+C1]==UInt8(49)
   a=sx-x[j+C1];b=sy-y[j+C1];sxy+=a*b;syy+=b*b
  else
   a=sn-x[j+C1];b=syn-y[j+C1];sxyn+=a*b;syyn+=b*b
  end
 end
 abs(sxy/(sxx*sqrt(syy))),abs(sxyn/(sxxn*sqrt(syyn)))
end
@inline function bicluster_body!(gene,n,m,D,thr,sample,data,score,dc,sc,tmps,tmpd,x,y,mask,k)
 @inbounds if k<m
  score[k+C1]=1f0;dc[k+C1]=C0;meank=gene[k*(D+C1)+D+C1]
  for i in Int32(k+C1):Int32(n-C1)
   meani=gene[i*(D+C1)+D+C1];w0=C0;w1=C0;w2=C0
   for j in C0:Int32(D-C1)
    a=gene[k*(D+C1)+j+C1]-meank;b=gene[i*(D+C1)+j+C1]-meani
    if a>=0f0&&b>=0f0
     mask[j+C1]=49;mask[L200+j+C1]=48;mask[2L200+j+C1]=48;w0+=C1
    elseif a<0f0&&b<0f0
     mask[j+C1]=48;mask[L200+j+C1]=49;mask[2L200+j+C1]=48;w1+=C1
    elseif a*b<0f0
     mask[j+C1]=48;mask[L200+j+C1]=48;mask[2L200+j+C1]=49;w2+=C1
    end
   end
   for vl in C0:Int32(2)
    wid=vl==0 ? w0 : vl==1 ? w1 : w2
    if wid>Int32(10)
     r,nr=correlation!(x,y,mask,vl*L200,wid,k,i,D,gene)
     if r>thr
      tot=C1;dif=Int32(nr>thr)
      for j in C0:Int32(D-C1);tmps[k*D+j+C1]=mask[vl*L200+j+C1];end
      for j in C0:Int32(n-C1);tmpd[k*n+j+C1]=48;end
      tmpd[k*n+k+C1]=49;tmpd[k*n+i+C1]=49;count=Int32(2)
      for l in C0:Int32(n-C1)
       if l!=i&&l!=k
        ttot=C0;tdif=C0
        for li in C0:Int32(n-C1)
         if tmpd[k*n+li+C1]==UInt8(49)
          rr,nn=correlation!(x,y,mask,vl*L200,wid,l,li,D,gene)
          if rr>thr;ttot+=C1;else;ttot=C0;break;end
          if nn>thr;tdif+=C1;end
         end
        end
        if ttot>0;tmpd[k*n+l+C1]=49;count+=C1;tot+=ttot;dif+=tdif;end
       end
      end
      jcc=tot>0 ? Float32(dif)/Float32(tot) : 1f0
      if jcc<.01f0&&dc[k+C1]<count&&count>Int32(10)
       score[k+C1]=jcc
       for j in C0:Int32(n-C1);data[k*n+j+C1]=tmpd[k*n+j+C1];end
       for j in C0:Int32(D-C1);sample[k*D+j+C1]=tmps[k*D+j+C1];end
       dc[k+C1]=count;sc[k+C1]=wid
      end
     end
    end
   end
  end
 end
 nothing
end
