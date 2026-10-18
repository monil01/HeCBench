@inline function back_site!(i,fb,lb,f,l,b,b3,e)
 if i<e
  for d in 0:3,r in 0:2,c in 0:2
   out=i*72+d*18+r*6+c*2+1
   j=Int(b[i*4+d+1])*72+d*18+c*6+r*2+1
   k=Int(b3[i*4+d+1])*72+d*18+c*6+r*2+1
   @inbounds fb[out]=f[j];fb[out+1]=-f[j+1];lb[out]=l[k];lb[out+1]=-l[k+1]
  end
 end
 nothing
end
@inline function dslash_site!(i,dst,s,f,l,fb,lb,fw,b,fw3,b3,e)
 if i<e
  for r in 0:2
   dr=0f0;di=0f0
   for group in 1:4
    tr=0f0;ti=0f0
    for d in 0:3
     mat=group==1 ? f : group==2 ? l : group==3 ? fb : lb
     mp=group==1 ? fw : group==2 ? fw3 : group==3 ? b : b3
     vr=0f0;vi=0f0
     @inbounds j=Int(mp[i*4+d+1])
     for c in 0:2
      a=i*72+d*18+r*6+c*2+1;q=j*6+c*2+1
      @inbounds ar=mat[a];ai=mat[a+1];br=s[q];bi=s[q+1]
      vr+=ar*br-ai*bi;vi+=ar*bi+ai*br
     end
     tr+=vr;ti+=vi
    end
    if group<=2;dr+=tr;di+=ti;else;dr-=tr;di-=ti;end
   end
   @inbounds dst[i*6+r*2+1]=dr;dst[i*6+r*2+2]=di
  end
 end
 nothing
end
