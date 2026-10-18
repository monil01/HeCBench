# CUDA-specific extension: actual host-initialized addrspace(4) constant symbol, SIMD vector loads.
using CUDA
@inline @generated function constant_load(::Val{W},idx::Int32) where W
 V=W==1 ? Int32 : NTuple{W,VecElement{Int32}};ty=W==1 ? "i32" : "<$(W) x i32>";a=W*4
 ir="""@study_constant_data = weak addrspace(4) externally_initialized global [1024 x i32] zeroinitializer, align 16
 define $ty @entry(i32 %idx) #0 {
 %base = mul i32 %idx, $W
 %wide = sext i32 %base to i64
 %p = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(4)* @study_constant_data, i64 0, i64 %wide
 %vp = bitcast i32 addrspace(4)* %p to $ty addrspace(4)*
 %v = load $ty, $ty addrspace(4)* %vp, align $a
 ret $ty %v
 }
 attributes #0 = { alwaysinline }
 """
 :(Base.llvmcall(($ir,"entry"),$V,Tuple{Int32},idx))
end
@inline @generated function constant_sum(::Val{W}) where W
 accum=[gensym(:sum) for  _ in 1:W];init=[:($(accum[k])=Int32(0)) for  k in 1:W];updates=[:($(accum[k])+=$(W==1 ? :v : :(v[$k].value))) for  k in 1:W];result=foldl((a,b)->:($a+$b),accum)
 quote
  $(init...)
  for i in Int32(0):Int32(3)
   j=Int32(0)
   while j<Int32($(1024÷W))
    v=constant_load(Val($W),j+i)
    $(updates...)
    j+=Int32(4)
   end
  end
  $result
 end
end
