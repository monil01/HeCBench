# Authored portable lane expansion of all118 source operations; no vendor calls.

@inline function simd_op_0(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=abs(x)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_1(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=abs(x)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_2(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=abs(x-y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_3(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=abs(x-y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_4(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=abs(x-y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_5(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=abs(x-y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_6(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(abs(x),Int32(32767))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_7(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=min(abs(x),Int32(127))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_8(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=x+y
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_9(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=x+y
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_10(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=clamp(x+y,Int32(-32768),Int32(32767))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_11(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=clamp(x+y,Int32(-128),Int32(127))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_12(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=clamp(x+y,Int32(0),Int32(65535))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_13(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=clamp(x+y,Int32(0),Int32(255))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_14(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=((x+y)>=0 ? (x+y+Int32(1)) : (x+y))>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_15(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=((x+y)>=0 ? (x+y+Int32(1)) : (x+y))>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_16(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=((x+y)>=0 ? (x+y+Int32(1)) : (x+y))>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_17(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=((x+y)>=0 ? (x+y+Int32(1)) : (x+y))>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_18(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x==y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_19(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x==y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_20(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x>=y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_21(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x>=y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_22(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>=y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_23(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>=y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_24(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x>y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_25(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x>y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_26(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_27(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_28(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x<=y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_29(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x<=y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_30(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<=y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_31(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<=y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_32(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x<y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_33(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x<y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_34(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_35(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_36(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x!=y) ? Int32(65535) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_37(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x!=y) ? Int32(255) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_38(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x+y)>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_39(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x+y)>>Int32(1)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_40(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  u=(x+y)&Int32(65535)
  u=u>=Int32(32768) ? u-Int32(65536) : u
  t=max(u,z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_41(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  u=(x+y)&Int32(65535)
  u=u>=Int32(32768) ? u-Int32(65536) : u
  t=max(u,z)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_42(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(reinterpret(Int32,a+b),z)
 return reinterpret(UInt32,t)
end

@inline function simd_op_43(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(reinterpret(Int32,a+b),z)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_44(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  u=(x+y)&Int32(65535)
  t=max(u,z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_45(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=max(a+b,z)
 return t
end

@inline function simd_op_46(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  u=(x+y)&Int32(65535)
  u=u>=Int32(32768) ? u-Int32(65536) : u
  t=min(u,z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_47(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  u=(x+y)&Int32(65535)
  u=u>=Int32(32768) ? u-Int32(65536) : u
  t=min(u,z)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_48(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(reinterpret(Int32,a+b),z)
 return reinterpret(UInt32,t)
end

@inline function simd_op_49(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(reinterpret(Int32,a+b),z)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_50(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  u=(x+y)&Int32(65535)
  t=min(u,z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_51(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=min(a+b,z)
 return t
end

@inline function simd_op_52(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_53(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(x,y)
 return reinterpret(UInt32,t)
end

@inline function simd_op_54(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_55(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=max(x,y)
 return t
end

@inline function simd_op_56(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_57(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(x,y)
 return reinterpret(UInt32,t)
end

@inline function simd_op_58(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_59(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=min(x,y)
 return t
end

@inline function simd_op_60(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=max(max(x,y),z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_61(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=max(max(x,y),z)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_62(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(max(x,y),z)
 return reinterpret(UInt32,t)
end

@inline function simd_op_63(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(max(x,y),z)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_64(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=max(max(x,y),z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_65(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=max(max(x,y),z)
 return t
end

@inline function simd_op_66(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=max(x,y)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_67(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=max(x,y)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_68(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(min(x,y),z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_69(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(min(x,y),z)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_70(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(min(x,y),z)
 return reinterpret(UInt32,t)
end

@inline function simd_op_71(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(min(x,y),z)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_72(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=min(min(x,y),z)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_73(a::UInt32,b::UInt32,c::UInt32)
 x=a;y=b;z=c
 t=min(min(x,y),z)
 return t
end

@inline function simd_op_74(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(x,y)
  t=max(t,Int32(0))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_75(a::UInt32,b::UInt32,c::UInt32)
 x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)
 t=min(x,y)
 t=max(t,Int32(0))
 return reinterpret(UInt32,t)
end

@inline function simd_op_76(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_77(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_78(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_79(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=max(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_80(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_81(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_82(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_83(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=min(x,y)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_84(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=-x
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_85(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=-x
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_86(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=min(-x,Int32(32767))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_87(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=min(-x,Int32(127))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_88(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=abs(x-y)
  result+=reinterpret(UInt32,t)
 end
 return result
end

@inline function simd_op_89(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=abs(x-y)
  result+=reinterpret(UInt32,t)
 end
 return result
end

@inline function simd_op_90(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=abs(x-y)
  result+=reinterpret(UInt32,t)
 end
 return result
end

@inline function simd_op_91(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=abs(x-y)
  result+=reinterpret(UInt32,t)
 end
 return result
end

@inline function simd_op_92(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x==y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_93(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x==y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_94(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x>=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_95(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x>=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_96(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_97(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_98(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x>y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_99(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x>y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_100(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_101(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x>y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_102(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x<=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_103(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x<=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_104(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_105(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_106(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=(x<y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_107(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=(x<y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_108(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_109(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x<y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_110(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x!=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_111(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=(x!=y) ? Int32(1) : Int32(0)
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_112(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=x-y
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_113(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=x-y
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_114(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(32768) ? x-Int32(65536) : x;y=y>=Int32(32768) ? y-Int32(65536) : y;z=z>=Int32(32768) ? z-Int32(65536) : z
  t=clamp(x-y,Int32(-32768),Int32(32767))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_115(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  x=x>=Int32(128) ? x-Int32(256) : x;y=y>=Int32(128) ? y-Int32(256) : y;z=z>=Int32(128) ? z-Int32(256) : z
  t=clamp(x-y,Int32(-128),Int32(127))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_116(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(65535);result=UInt32(0)
 for lane in Int32(0):Int32(1)
  shift=lane*Int32(16);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=clamp(x-y,Int32(0),Int32(65535))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_op_117(a::UInt32,b::UInt32,c::UInt32)
 mask=UInt32(255);result=UInt32(0)
 for lane in Int32(0):Int32(3)
  shift=lane*Int32(8);x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)
  t=clamp(x-y,Int32(0),Int32(255))
  result|=(reinterpret(UInt32,t)&mask)<<shift
 end
 return result
end

@inline function simd_xor(a::UInt32,b::UInt32,c::UInt32)
 r=simd_op_0(a,b,c)
 r⊻=simd_op_1(a,b,c)
 r⊻=simd_op_2(a,b,c)
 r⊻=simd_op_3(a,b,c)
 r⊻=simd_op_4(a,b,c)
 r⊻=simd_op_5(a,b,c)
 r⊻=simd_op_6(a,b,c)
 r⊻=simd_op_7(a,b,c)
 r⊻=simd_op_8(a,b,c)
 r⊻=simd_op_9(a,b,c)
 r⊻=simd_op_10(a,b,c)
 r⊻=simd_op_11(a,b,c)
 r⊻=simd_op_12(a,b,c)
 r⊻=simd_op_13(a,b,c)
 r⊻=simd_op_14(a,b,c)
 r⊻=simd_op_15(a,b,c)
 r⊻=simd_op_16(a,b,c)
 r⊻=simd_op_17(a,b,c)
 r⊻=simd_op_18(a,b,c)
 r⊻=simd_op_19(a,b,c)
 r⊻=simd_op_20(a,b,c)
 r⊻=simd_op_21(a,b,c)
 r⊻=simd_op_22(a,b,c)
 r⊻=simd_op_23(a,b,c)
 r⊻=simd_op_24(a,b,c)
 r⊻=simd_op_25(a,b,c)
 r⊻=simd_op_26(a,b,c)
 r⊻=simd_op_27(a,b,c)
 r⊻=simd_op_28(a,b,c)
 r⊻=simd_op_29(a,b,c)
 r⊻=simd_op_30(a,b,c)
 r⊻=simd_op_31(a,b,c)
 r⊻=simd_op_32(a,b,c)
 r⊻=simd_op_33(a,b,c)
 r⊻=simd_op_34(a,b,c)
 r⊻=simd_op_35(a,b,c)
 r⊻=simd_op_36(a,b,c)
 r⊻=simd_op_37(a,b,c)
 r⊻=simd_op_38(a,b,c)
 r⊻=simd_op_39(a,b,c)
 r⊻=simd_op_40(a,b,c)
 r⊻=simd_op_41(a,b,c)
 r⊻=simd_op_42(a,b,c)
 r⊻=simd_op_43(a,b,c)
 r⊻=simd_op_44(a,b,c)
 r⊻=simd_op_45(a,b,c)
 r⊻=simd_op_46(a,b,c)
 r⊻=simd_op_47(a,b,c)
 r⊻=simd_op_48(a,b,c)
 r⊻=simd_op_49(a,b,c)
 r⊻=simd_op_50(a,b,c)
 r⊻=simd_op_51(a,b,c)
 r⊻=simd_op_52(a,b,c)
 r⊻=simd_op_53(a,b,c)
 r⊻=simd_op_54(a,b,c)
 r⊻=simd_op_55(a,b,c)
 r⊻=simd_op_56(a,b,c)
 r⊻=simd_op_57(a,b,c)
 r⊻=simd_op_58(a,b,c)
 r⊻=simd_op_59(a,b,c)
 r⊻=simd_op_60(a,b,c)
 r⊻=simd_op_61(a,b,c)
 r⊻=simd_op_62(a,b,c)
 r⊻=simd_op_63(a,b,c)
 r⊻=simd_op_64(a,b,c)
 r⊻=simd_op_65(a,b,c)
 r⊻=simd_op_66(a,b,c)
 r⊻=simd_op_67(a,b,c)
 r⊻=simd_op_68(a,b,c)
 r⊻=simd_op_69(a,b,c)
 r⊻=simd_op_70(a,b,c)
 r⊻=simd_op_71(a,b,c)
 r⊻=simd_op_72(a,b,c)
 r⊻=simd_op_73(a,b,c)
 r⊻=simd_op_74(a,b,c)
 r⊻=simd_op_75(a,b,c)
 r⊻=simd_op_76(a,b,c)
 r⊻=simd_op_77(a,b,c)
 r⊻=simd_op_78(a,b,c)
 r⊻=simd_op_79(a,b,c)
 r⊻=simd_op_80(a,b,c)
 r⊻=simd_op_81(a,b,c)
 r⊻=simd_op_82(a,b,c)
 r⊻=simd_op_83(a,b,c)
 r⊻=simd_op_84(a,b,c)
 r⊻=simd_op_85(a,b,c)
 r⊻=simd_op_86(a,b,c)
 r⊻=simd_op_87(a,b,c)
 r⊻=simd_op_88(a,b,c)
 r⊻=simd_op_89(a,b,c)
 r⊻=simd_op_90(a,b,c)
 r⊻=simd_op_91(a,b,c)
 r⊻=simd_op_92(a,b,c)
 r⊻=simd_op_93(a,b,c)
 r⊻=simd_op_94(a,b,c)
 r⊻=simd_op_95(a,b,c)
 r⊻=simd_op_96(a,b,c)
 r⊻=simd_op_97(a,b,c)
 r⊻=simd_op_98(a,b,c)
 r⊻=simd_op_99(a,b,c)
 r⊻=simd_op_100(a,b,c)
 r⊻=simd_op_101(a,b,c)
 r⊻=simd_op_102(a,b,c)
 r⊻=simd_op_103(a,b,c)
 r⊻=simd_op_104(a,b,c)
 r⊻=simd_op_105(a,b,c)
 r⊻=simd_op_106(a,b,c)
 r⊻=simd_op_107(a,b,c)
 r⊻=simd_op_108(a,b,c)
 r⊻=simd_op_109(a,b,c)
 r⊻=simd_op_110(a,b,c)
 r⊻=simd_op_111(a,b,c)
 r⊻=simd_op_112(a,b,c)
 r⊻=simd_op_113(a,b,c)
 r⊻=simd_op_114(a,b,c)
 r⊻=simd_op_115(a,b,c)
 r⊻=simd_op_116(a,b,c)
 r⊻=simd_op_117(a,b,c)
 return r
end

@inline function individual_operation(op::Int32,a::UInt32,b::UInt32,c::UInt32)
 if op==Int32(0)
  return simd_op_0(a,b,c)
 elseif op==Int32(1)
  return simd_op_1(a,b,c)
 elseif op==Int32(2)
  return simd_op_2(a,b,c)
 elseif op==Int32(3)
  return simd_op_3(a,b,c)
 elseif op==Int32(4)
  return simd_op_4(a,b,c)
 elseif op==Int32(5)
  return simd_op_5(a,b,c)
 elseif op==Int32(6)
  return simd_op_6(a,b,c)
 elseif op==Int32(7)
  return simd_op_7(a,b,c)
 elseif op==Int32(8)
  return simd_op_8(a,b,c)
 elseif op==Int32(9)
  return simd_op_9(a,b,c)
 elseif op==Int32(10)
  return simd_op_10(a,b,c)
 elseif op==Int32(11)
  return simd_op_11(a,b,c)
 elseif op==Int32(12)
  return simd_op_12(a,b,c)
 elseif op==Int32(13)
  return simd_op_13(a,b,c)
 elseif op==Int32(14)
  return simd_op_14(a,b,c)
 elseif op==Int32(15)
  return simd_op_15(a,b,c)
 elseif op==Int32(16)
  return simd_op_16(a,b,c)
 elseif op==Int32(17)
  return simd_op_17(a,b,c)
 elseif op==Int32(18)
  return simd_op_18(a,b,c)
 elseif op==Int32(19)
  return simd_op_19(a,b,c)
 elseif op==Int32(20)
  return simd_op_20(a,b,c)
 elseif op==Int32(21)
  return simd_op_21(a,b,c)
 elseif op==Int32(22)
  return simd_op_22(a,b,c)
 elseif op==Int32(23)
  return simd_op_23(a,b,c)
 elseif op==Int32(24)
  return simd_op_24(a,b,c)
 elseif op==Int32(25)
  return simd_op_25(a,b,c)
 elseif op==Int32(26)
  return simd_op_26(a,b,c)
 elseif op==Int32(27)
  return simd_op_27(a,b,c)
 elseif op==Int32(28)
  return simd_op_28(a,b,c)
 elseif op==Int32(29)
  return simd_op_29(a,b,c)
 elseif op==Int32(30)
  return simd_op_30(a,b,c)
 elseif op==Int32(31)
  return simd_op_31(a,b,c)
 elseif op==Int32(32)
  return simd_op_32(a,b,c)
 elseif op==Int32(33)
  return simd_op_33(a,b,c)
 elseif op==Int32(34)
  return simd_op_34(a,b,c)
 elseif op==Int32(35)
  return simd_op_35(a,b,c)
 elseif op==Int32(36)
  return simd_op_36(a,b,c)
 elseif op==Int32(37)
  return simd_op_37(a,b,c)
 elseif op==Int32(38)
  return simd_op_38(a,b,c)
 elseif op==Int32(39)
  return simd_op_39(a,b,c)
 elseif op==Int32(40)
  return simd_op_40(a,b,c)
 elseif op==Int32(41)
  return simd_op_41(a,b,c)
 elseif op==Int32(42)
  return simd_op_42(a,b,c)
 elseif op==Int32(43)
  return simd_op_43(a,b,c)
 elseif op==Int32(44)
  return simd_op_44(a,b,c)
 elseif op==Int32(45)
  return simd_op_45(a,b,c)
 elseif op==Int32(46)
  return simd_op_46(a,b,c)
 elseif op==Int32(47)
  return simd_op_47(a,b,c)
 elseif op==Int32(48)
  return simd_op_48(a,b,c)
 elseif op==Int32(49)
  return simd_op_49(a,b,c)
 elseif op==Int32(50)
  return simd_op_50(a,b,c)
 elseif op==Int32(51)
  return simd_op_51(a,b,c)
 elseif op==Int32(52)
  return simd_op_52(a,b,c)
 elseif op==Int32(53)
  return simd_op_53(a,b,c)
 elseif op==Int32(54)
  return simd_op_54(a,b,c)
 elseif op==Int32(55)
  return simd_op_55(a,b,c)
 elseif op==Int32(56)
  return simd_op_56(a,b,c)
 elseif op==Int32(57)
  return simd_op_57(a,b,c)
 elseif op==Int32(58)
  return simd_op_58(a,b,c)
 elseif op==Int32(59)
  return simd_op_59(a,b,c)
 elseif op==Int32(60)
  return simd_op_60(a,b,c)
 elseif op==Int32(61)
  return simd_op_61(a,b,c)
 elseif op==Int32(62)
  return simd_op_62(a,b,c)
 elseif op==Int32(63)
  return simd_op_63(a,b,c)
 elseif op==Int32(64)
  return simd_op_64(a,b,c)
 elseif op==Int32(65)
  return simd_op_65(a,b,c)
 elseif op==Int32(66)
  return simd_op_66(a,b,c)
 elseif op==Int32(67)
  return simd_op_67(a,b,c)
 elseif op==Int32(68)
  return simd_op_68(a,b,c)
 elseif op==Int32(69)
  return simd_op_69(a,b,c)
 elseif op==Int32(70)
  return simd_op_70(a,b,c)
 elseif op==Int32(71)
  return simd_op_71(a,b,c)
 elseif op==Int32(72)
  return simd_op_72(a,b,c)
 elseif op==Int32(73)
  return simd_op_73(a,b,c)
 elseif op==Int32(74)
  return simd_op_74(a,b,c)
 elseif op==Int32(75)
  return simd_op_75(a,b,c)
 elseif op==Int32(76)
  return simd_op_76(a,b,c)
 elseif op==Int32(77)
  return simd_op_77(a,b,c)
 elseif op==Int32(78)
  return simd_op_78(a,b,c)
 elseif op==Int32(79)
  return simd_op_79(a,b,c)
 elseif op==Int32(80)
  return simd_op_80(a,b,c)
 elseif op==Int32(81)
  return simd_op_81(a,b,c)
 elseif op==Int32(82)
  return simd_op_82(a,b,c)
 elseif op==Int32(83)
  return simd_op_83(a,b,c)
 elseif op==Int32(84)
  return simd_op_84(a,b,c)
 elseif op==Int32(85)
  return simd_op_85(a,b,c)
 elseif op==Int32(86)
  return simd_op_86(a,b,c)
 elseif op==Int32(87)
  return simd_op_87(a,b,c)
 elseif op==Int32(88)
  return simd_op_88(a,b,c)
 elseif op==Int32(89)
  return simd_op_89(a,b,c)
 elseif op==Int32(90)
  return simd_op_90(a,b,c)
 elseif op==Int32(91)
  return simd_op_91(a,b,c)
 elseif op==Int32(92)
  return simd_op_92(a,b,c)
 elseif op==Int32(93)
  return simd_op_93(a,b,c)
 elseif op==Int32(94)
  return simd_op_94(a,b,c)
 elseif op==Int32(95)
  return simd_op_95(a,b,c)
 elseif op==Int32(96)
  return simd_op_96(a,b,c)
 elseif op==Int32(97)
  return simd_op_97(a,b,c)
 elseif op==Int32(98)
  return simd_op_98(a,b,c)
 elseif op==Int32(99)
  return simd_op_99(a,b,c)
 elseif op==Int32(100)
  return simd_op_100(a,b,c)
 elseif op==Int32(101)
  return simd_op_101(a,b,c)
 elseif op==Int32(102)
  return simd_op_102(a,b,c)
 elseif op==Int32(103)
  return simd_op_103(a,b,c)
 elseif op==Int32(104)
  return simd_op_104(a,b,c)
 elseif op==Int32(105)
  return simd_op_105(a,b,c)
 elseif op==Int32(106)
  return simd_op_106(a,b,c)
 elseif op==Int32(107)
  return simd_op_107(a,b,c)
 elseif op==Int32(108)
  return simd_op_108(a,b,c)
 elseif op==Int32(109)
  return simd_op_109(a,b,c)
 elseif op==Int32(110)
  return simd_op_110(a,b,c)
 elseif op==Int32(111)
  return simd_op_111(a,b,c)
 elseif op==Int32(112)
  return simd_op_112(a,b,c)
 elseif op==Int32(113)
  return simd_op_113(a,b,c)
 elseif op==Int32(114)
  return simd_op_114(a,b,c)
 elseif op==Int32(115)
  return simd_op_115(a,b,c)
 elseif op==Int32(116)
  return simd_op_116(a,b,c)
 elseif op==Int32(117)
  return simd_op_117(a,b,c)
 end
 return UInt32(0) # unreachable: everydeclared probe op is0..117
end
