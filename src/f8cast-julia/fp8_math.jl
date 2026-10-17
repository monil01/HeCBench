# Original source bit-encoder algorithm; helper authored once in reference preparation.
@inline function fp8_ref(x::Float32,fmt::Int32)
 b=reinterpret(UInt32,x);sign=b&0x80000000;b=b⊻sign
 if fmt==Int32(0)
  if b>=UInt32(1087)<<20
   r=UInt32(0x7f)
  elseif b<UInt32(121)<<23
   magic=UInt32(141)<<23;r=reinterpret(UInt32,reinterpret(Float32,b)+reinterpret(Float32,magic))-magic
  else
   odd=(b>>20)&UInt32(1);b+=UInt32(0xc4000000)+UInt32(0x7ffff)+odd;r=b>>20
  end
 else
  if b>=UInt32(143)<<23
   r=b>UInt32(255)<<23 ? UInt32(0x7f) : UInt32(0x7c)
  elseif b<UInt32(113)<<23
   magic=UInt32(134)<<23;r=reinterpret(UInt32,reinterpret(Float32,b)+reinterpret(Float32,magic))-magic
  else
   odd=(b>>21)&UInt32(1);b+=UInt32(0xc8000000)+UInt32(0xfffff)+odd;r=b>>21
  end
 end
 return UInt8((r|(sign>>24))&UInt32(255))
end
@inline function fp8_sat(x::Float32,fmt::Int32)
 b=reinterpret(UInt32,x);mag=b&0x7fffffff
 if mag>UInt32(0x7f800000);return UInt8(0x7f);end
 maxbits=fmt==Int32(0) ? reinterpret(UInt32,448f0) : reinterpret(UInt32,57344f0)
 clamped=reinterpret(Float32,(b&0x80000000)|min(mag,maxbits))
 return fp8_ref(clamped,fmt)
end
