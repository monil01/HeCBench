# Version-compatible FP32 approximation reconstructed from the actual unchanged
# native CUDA12.0 NVCC PTX for source atan(float). Deliberate rounding policy:
# rn reciprocal and each displayed FMA; not a general higher-accuracy claim.
@inline function source_atanf(x::Float32)
 a=abs(x);t=a>1.0f0 ? 1.0f0/a : a;z=t*t
 p=muladd(z,reinterpret(Float32,0xbf52c7ea),reinterpret(Float32,0xc0b59883))
 p=muladd(p,z,reinterpret(Float32,0xc0d21907));num=t*(z*p)
 den=muladd(z+reinterpret(Float32,0x41355dc0),z,reinterpret(Float32,0x41e6bd60))
 den=muladd(den,z,reinterpret(Float32,0x419d92c8));value=muladd(num,1.0f0/den,t)
 if a>1.0f0;value=reinterpret(Float32,0x3fc90fdb)-value;end
 a<=Inf32 ? copysign(value,x) : value
end
