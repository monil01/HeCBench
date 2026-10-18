@inline function encode_e4m3(x::Float32)
 b=reinterpret(UInt32,x);sg=UInt8((b>>24)&0x80);a=b&0x7fffffff
 if a>=0x43e00000;return sg|0x7e;end
 e=a>>23
 if e<117;return sg
 elseif e<121
  mant=(a&0x7fffff)|0x800000;shift=UInt32(141)-e;code=mant>>shift;rem=mant&((UInt32(1)<<shift)-1);half=UInt32(1)<<(shift-1)
  code+=UInt32(rem>half||(rem==half&&isodd(code)))
 else
  code=(a>>20)-UInt32(960);rem=a&0xfffff;code+=UInt32(rem>0x80000||(rem==0x80000&&isodd(code)))
 end
 return sg|UInt8(code&0x7f)
end
@inline function cache_lane(token,tid,k,v,kc,vc,slots,H,D,B,x,ks,vs)
 slot=@inbounds slots[token+1]
 if slot>=0
  blocks=D÷x;hb=tid
  while hb<H*blocks
   head=hb÷blocks;db=hb%blocks;src=token*H*D+head*D+db*x;dst=(((slot÷B*H+head)*blocks+db)*B+slot%B)*x;vd=((slot÷B*H+head)*D+db*x)*B+slot%B
   if x>=4
    kp=reinterpret(NTuple{4,Float32},k);cp=reinterpret(NTuple{4,UInt8},kc)
    j=Int32(0)
    while j<x
     a=@inbounds kp[(src+j)÷4+1];q=(encode_e4m3(a[1]/ks),encode_e4m3(a[2]/ks),encode_e4m3(a[3]/ks),encode_e4m3(a[4]/ks));@inbounds cp[(dst+j)÷4+1]=q;j+=4
    end
   else
    j=Int32(0);while j<x;@inbounds kc[dst+j+1]=encode_e4m3(k[src+j+1]/ks);j+=1;end
   end
   j=Int32(0);while j<x;@inbounds vc[vd+j*B+1]=encode_e4m3(v[src+j+1]/vs);j+=1;end
   hb+=Int32(512)
  end
 end
 nothing
end
