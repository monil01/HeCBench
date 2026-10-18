# Source KISS64/Murmur semantics translated from kiss.cuh/hashers.cuh.
# Copyright2022 Daniel Jünger, Apache2.0; portable Julia adaptation2026.
@inline function murmur32(x::UInt32)
 x⊻=x>>16;x*=UInt32(0x85ebca6b);x⊻=x>>13;x*=UInt32(0xc2b2ae35);x⊻=x>>16;return x
end
@inline function murmur64(x::UInt64)
 x⊻=x>>33;x*=UInt64(0xff51afd7ed558ccd);x⊻=x>>33;x*=UInt64(0xc4ceb9fe1a85ec53);x⊻=x>>33;return x
end
@inline function kiss_body!(out,n::UInt64,seed::UInt32,tid::UInt32,stride::UInt64)
 local_seed=murmur32(seed+tid);w=iszero(local_seed) ? UInt64(4294967295) : UInt64(local_seed);x=UInt64(0);y=UInt64(0);z=UInt64(0)
 for _ in Int32(1):Int32(8)
  x=murmur64(w);y=murmur64(x);z=murmur64(y);w=murmur64(z)
 end
 # Murmur is bijective and nonzero fallback guarantees these conditions globally.
 # Retain a diagnostic branch; invalidstate skips writes and verificationfails rather than CUDAassertinterop.
 if iszero(y)||iszero(z)||iszero(w);return nothing;end
 i=UInt64(tid)
 while i<n
  x=UInt64(6906969069)*x+UInt64(1234567)
  y⊻=y<<13;y⊻=y>>17;y⊻=y<<43
  t=(z<<58)+w;w=z>>6;z+=t;w+=UInt64(z<t)
  @inbounds out[i+UInt64(1)]=x+y+z
  i+=stride
 end
 return nothing
end
