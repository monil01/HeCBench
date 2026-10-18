@inline function packed_add(a::UInt32,b::UInt32)
 CUDA.CUDACore.@asmcall("add.rn.f16x2 \$0, \$1, \$2;", "=r,r,r", false, UInt32, Tuple{UInt32,UInt32}, a,b)
end
@inline function packed_fma(a::UInt32,b::UInt32,c::UInt32)
 CUDA.CUDACore.@asmcall("fma.rn.f16x2 \$0, \$1, \$2, \$3;", "=r,r,r,r", false, UInt32, Tuple{UInt32,UInt32,UInt32}, a,b,c)
end
@generated function arithmetic_chain(x::T) where T
 if T===UInt32
  ops=[:(beta=packed_fma(beta,x,UInt32(0x40004000))) for _ in 1:511]
  return quote
   beta=packed_add(x,UInt32(0x40004000))
   $(ops...)
   beta ⊻ UInt32(0x80008000)
  end
 else
  ops=[:(beta=muladd(beta,x,T(2))) for _ in 1:1023]
  return quote
   beta=x+T(2)
   $(ops...)
   -beta
  end
 end
end
@inline function ert_site!(start,total,A,n,trials)
 elems=cld(n,total);finish=min(start+elems*total,n);first=min(start,n)
 for j in UInt32(1):trials
  i=first
  while i<finish
   @inbounds A[Int(i)+1]=arithmetic_chain(A[Int(i)+1]);i+=total
  end
 end
 nothing
end
