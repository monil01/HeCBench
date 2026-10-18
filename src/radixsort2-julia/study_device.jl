@inline ordered_bits(x::UInt32)=x
# SourceFloat32inputisnonnegativefinite; rawbitorderisnumericalorder inthisdomain.
@inline ordered_bits(x::Float32)=reinterpret(UInt32,x)
@inline function radix_destination(i,key,localzero,blockoffset,totalzero,bit)
 @inbounds begin
  z=localzero[i+Int32(1)]+blockoffset[(i>>Int32(10))+Int32(1)]
  ((ordered_bits(key)>>bit)&UInt32(1))==0 ? z : totalzero[1]+UInt32(i)-z
 end
end
