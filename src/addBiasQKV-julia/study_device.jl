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

@inline function bfround(x::Float32)
 u=reinterpret(UInt32,x);r=u+UInt32(0x7fff)+((u>>16)&UInt32(1));reinterpret(Float32,r&UInt32(0xffff0000))
end
@inline function decode_e4m3(v::UInt8)
 s=UInt32(v&0x80)<<24;e=UInt32((v>>3)&0x0f);m=UInt32(v&7)
 if e==0;return copysign(Float32(m)*Float32(0.001953125),reinterpret(Float32,s));end
 reinterpret(Float32,s|((e+UInt32(120))<<23)|(m<<20))
end
@inline function transform_fp8(v::UInt8,b::UInt16,si::Float32,so::Float32)
 x=bfround(decode_e4m3(v));x=bfround(x*bfround(si));x=bfround(x+reinterpret(Float32,UInt32(b)<<16));x=bfround(x*bfround(so));encode_e4m3(x)
end
@inline function qkv_lane(token,t,src,bias,out,scale,H,D)
 size4=D÷Int32(4);hidden4=H*size4;x=t%size4;h=(t÷size4)%H;c=t÷hidden4;sid=c*hidden4+h*size4+x;oid=h*Int32(3)*size4+c*size4+x
 sp=reinterpret(NTuple{4,UInt8},src);op=reinterpret(NTuple{4,UInt8},out)
 @inbounds begin
 # Source/output prefix guards64bytes; bias prefix32UInt16words.
 a=sp[token*Int32(3)*hidden4+sid+Int32(17)];bi=sid*Int32(4)+Int32(33);si=scale[1];so=scale[2]
 q=(transform_fp8(a[1],bias[bi],si,so),transform_fp8(a[2],bias[bi+1],si,so),transform_fp8(a[3],bias[bi+2],si,so),transform_fp8(a[4],bias[bi+3],si,so));op[token*Int32(3)*hidden4+oid+Int32(17)]=q
 end
 nothing
end
