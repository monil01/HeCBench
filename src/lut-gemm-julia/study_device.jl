# Source mapping of NAVER LUT-GEMM mv_fp16.h selected K_TILE=32/64.
# Genuine FP16 operations and packed FP16x2 atomic: CUDA backend extension.
@inline function h_add(a::Float16,b::Float16)
 reinterpret(Float16,CUDA.CUDACore.@asmcall("add.f16 \$0,\$1,\$2;","=h,h,h",false,UInt16,Tuple{UInt16,UInt16},reinterpret(UInt16,a),reinterpret(UInt16,b)))
end
@inline function h_mul(a::Float16,b::Float16)
 reinterpret(Float16,CUDA.CUDACore.@asmcall("mul.f16 \$0,\$1,\$2;","=h,h,h",false,UInt16,Tuple{UInt16,UInt16},reinterpret(UInt16,a),reinterpret(UInt16,b)))
end
@inline function h_fma(a::Float16,b::Float16,c::Float16)
 reinterpret(Float16,CUDA.CUDACore.@asmcall("fma.rn.f16 \$0,\$1,\$2,\$3;","=h,h,h,h",false,UInt16,Tuple{UInt16,UInt16,UInt16},reinterpret(UInt16,a),reinterpret(UInt16,b),reinterpret(UInt16,c)))
end
@inline function packed_add!(out,m,a,b)
 ptr=reinterpret(Core.LLVMPtr{UInt32,1},pointer(out,m+1));v=UInt32(reinterpret(UInt16,a))|(UInt32(reinterpret(UInt16,b))<<16)
 CUDA.CUDACore.@asmcall("atom.add.noftz.f16x2 \$0,[\$1],\$2;","=r,l,r",true,UInt32,Tuple{Core.LLVMPtr{UInt32,1},UInt32},ptr,v);nothing
end
@inline function lut_build!(lut,X,t,by,::Val{T}) where T
 nx=256÷(T÷8);row=t÷nx;col=t%nx;base=Float16(0)
 @inbounds for b in 0:7
  sign=Float16(2*((col>>b)&1)-1);base=h_fma(sign,X[by*T+row*8+b+1],base)
 end
 @inbounds lut[row*256+col+1]=base
 s= T==32 ? 6 : 5
 @inbounds for bit in s:7
  v=h_mul(Float16(2),X[by*T+row*8+bit+1]);i=1<<bit
  while i<(1<<(bit+1))
   lut[row*256+i+col+1]=h_add(lut[row*256+i+col-(1<<bit)+1],v);i+=nx
  end
 end
 nothing
end
@inline function lut_finish!(lut,W,A,B,X,Y,M,K,nb,G,t,bx,by,::Val{T},::Val{BIAS}) where {T,BIAS}
 m=bx*2048+t*2;stop=min((bx+1)*2048,M);g=(by*T)÷G
 @inbounds while m<stop
  o0=Float16(0);o1=Float16(0)
  if BIAS
   s=Float16(0)
   for row in 0:(T÷8-1);s=h_add(s,lut[row*256+256]);end
   o0=h_mul(s,B[g*M+m+1]);o1=h_mul(s,B[g*M+m+2])
  end
  for bit in 0:nb-1
   s0=Float16(0);s1=Float16(0)
   for kt in 0:(T÷32-1)
    wi=((by*T÷32+kt)*nb+bit)*M+m
    w0=W[wi+1];w1=W[wi+2]
    for byte in 0:3
     s0=h_add(s0,lut[(kt*4+byte)*256+Int((w0>>(8*byte))&255)+1]);s1=h_add(s1,lut[(kt*4+byte)*256+Int((w1>>(8*byte))&255)+1])
    end
   end
   o0=h_fma(A[(g*nb+bit)*M+m+1],s0,o0);o1=h_fma(A[(g*nb+bit)*M+m+2],s1,o1)
  end
  packed_add!(Y,m,o0,o1);m+=512
 end
 nothing
end
