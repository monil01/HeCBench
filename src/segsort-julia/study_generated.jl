
# Mechanically translated generated CUDA networks: C spans retain zero-based indexing/pointer offsets.
struct CSpan{A}
 data::A
 offset::Int32
end
@inline Base.getindex(a::CSpan,i::Integer)=@inbounds a.data[a.offset+i+1]
@inline Base.setindex!(a::CSpan,v,i::Integer)=(@inbounds a.data[a.offset+i+1]=v)
@inline Base.:+(a::CSpan,i::Integer)=CSpan(a.data,a.offset+Int32(i))
@inline sg_pointer(a::CSpan,i::Integer)=pointer(a.data,a.offset+i+1)

# bb_comput_common.cuh 9d8aed57ed1035d855499b802576b1ff041e556037a57ea467dd9afd9b9b2298
@inline function sg_upper_power_of_two(ctx,v_v)
 @inbounds begin
 v_v -= Int32(1)
 v_v |= (v_v >> Int32(1))
 v_v |= (v_v >> Int32(2))
 v_v |= (v_v >> Int32(4))
 v_v |= (v_v >> Int32(8))
 v_v |= (v_v >> Int32(16))
 v_v += Int32(1)
 return v_v
 end
end

# bb_comput_common.cuh 9d8aed57ed1035d855499b802576b1ff041e556037a57ea467dd9afd9b9b2298
@inline function sg_log2(ctx,v_u)
 @inbounds begin
 local v_s::Int32 = zero(Int32)
 local v_t::Int32 = zero(Int32)
 v_t = ((v_u > Int32(0xffff)) << Int32(4))
 v_u >>= v_t
 v_s = ((v_u > Int32(0xff)) << Int32(3))
 v_u >>= v_s;v_t |= v_s
 v_s = ((v_u > Int32(0xf)) << Int32(2))
 v_u >>= v_s;v_t |= v_s
 v_s = ((v_u > Int32(0x3)) << Int32(1))
 v_u >>= v_s;v_t |= v_s
 return (v_t | (v_u >> Int32(1)))
 end
end

# bb_comput_common.cuh 9d8aed57ed1035d855499b802576b1ff041e556037a57ea467dd9afd9b9b2298
@inline function sg_find_kth3(ctx,v_a,v_aCount,v_b,v_bCount,v_diag)
 @inbounds begin
 local v_begin::Int32 = zero(Int32)
 local v_end::Int32 = zero(Int32)
 local v_mid::Int32 = zero(Int32)
 local v_aKey::Int32 = zero(Int32)
 local v_bKey::Int32 = zero(Int32)
 local v_pred::Bool = zero(Bool)
 v_begin = max(Int32(0),(v_diag - v_bCount))
 v_end = min(v_diag,v_aCount)
 while (v_begin < v_end)
  v_mid = ((v_begin + v_end) >> Int32(1))
  v_aKey = v_a[v_mid]
  v_bKey = v_b[((v_diag - Int32(1)) - v_mid)]
  v_pred = (v_aKey <= v_bKey)
  if (v_pred != 0)
   v_begin = (v_mid + Int32(1))
  else
   v_end = v_mid
  end
 end
 return v_begin
 end
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k0,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 end
 return (v_k0,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_k1,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 end
 return (v_k0,v_k1,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k2
  v_k2 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k3
  v_k3 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k2
  v_k2 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k3
  v_k3 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k6
  v_k6 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k4
  v_k4 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k5
  v_k5 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k6
  v_k6 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k4
  v_k4 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k5
  v_k5 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k14
  v_k14 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k15
  v_k15 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k12
  v_k12 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k10
  v_k10 = v__t
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k8
  v_k8 = v__t
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k9
  v_k9 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k14
  v_k14 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k15
  v_k15 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k12
  v_k12 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k10
  v_k10 = v__t
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k8
  v_k8 = v__t
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k9
  v_k9 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_intxn_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k30
  v_k30 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k31
  v_k31 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k28
  v_k28 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k29
  v_k29 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k26
  v_k26 = v__t
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k27
  v_k27 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k24
  v_k24 = v__t
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k25
  v_k25 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k22
  v_k22 = v__t
 end
 if (v_bit != 0)
  v__t = v_k9
  v_k9 = v_k23
  v_k23 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k20
  v_k20 = v__t
 end
 if (v_bit != 0)
  v__t = v_k11
  v_k11 = v_k21
  v_k21 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k18
  v_k18 = v__t
 end
 if (v_bit != 0)
  v__t = v_k13
  v_k13 = v_k19
  v_k19 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k16
  v_k16 = v__t
 end
 if (v_bit != 0)
  v__t = v_k15
  v_k15 = v_k17
  v_k17 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k16
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k17,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k16 = v_ex_k0
 v_k17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k18
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k19,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k18 = v_ex_k0
 v_k19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k20
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k21,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k20 = v_ex_k0
 v_k21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k22
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k23,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k22 = v_ex_k0
 v_k23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k24
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k25,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k24 = v_ex_k0
 v_k25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k26
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k27,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k26 = v_ex_k0
 v_k27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k28
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k29,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k28 = v_ex_k0
 v_k29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k30
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k31,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k30 = v_ex_k0
 v_k31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k30
  v_k30 = v__t
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k31
  v_k31 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k28
  v_k28 = v__t
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k29
  v_k29 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k26
  v_k26 = v__t
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k27
  v_k27 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k24
  v_k24 = v__t
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k25
  v_k25 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k22
  v_k22 = v__t
 end
 if (v_bit != 0)
  v__t = v_k9
  v_k9 = v_k23
  v_k23 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k20
  v_k20 = v__t
 end
 if (v_bit != 0)
  v__t = v_k11
  v_k11 = v_k21
  v_k21 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k18
  v_k18 = v__t
 end
 if (v_bit != 0)
  v__t = v_k13
  v_k13 = v_k19
  v_k19 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k16
  v_k16 = v__t
 end
 if (v_bit != 0)
  v__t = v_k15
  v_k15 = v_k17
  v_k17 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k0,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 end
 return (v_k0,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_k1,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 end
 return (v_k0,v_k1,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,)
end

# bb_exch_keys.cuh 33507d22a4ce043e427a2697b3e42700867fff3eaa1ace0574845ad19c8d5654
@inline function sg_exch_paral_keys(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
 end
 if (v_bit != 0)
  v__t = v_k16
  v_k16 = v_k17
  v_k17 = v__t
 end
 if (v_bit != 0)
  v__t = v_k18
  v_k18 = v_k19
  v_k19 = v__t
 end
 if (v_bit != 0)
  v__t = v_k20
  v_k20 = v_k21
  v_k21 = v__t
 end
 if (v_bit != 0)
  v__t = v_k22
  v_k22 = v_k23
  v_k23 = v__t
 end
 if (v_bit != 0)
  v__t = v_k24
  v_k24 = v_k25
  v_k25 = v__t
 end
 if (v_bit != 0)
  v__t = v_k26
  v_k26 = v_k27
  v_k27 = v__t
 end
 if (v_bit != 0)
  v__t = v_k28
  v_k28 = v_k29
  v_k29 = v__t
 end
 if (v_bit != 0)
  v__t = v_k30
  v_k30 = v_k31
  v_k31 = v__t
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k16
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k17,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k16 = v_ex_k0
 v_k17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k18
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k19,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k18 = v_ex_k0
 v_k19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k20
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k21,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k20 = v_ex_k0
 v_k21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k22
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k23,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k22 = v_ex_k0
 v_k23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k24
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k25,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k24 = v_ex_k0
 v_k25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k26
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k27,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k26 = v_ex_k0
 v_k27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k28
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k29,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k28 = v_ex_k0
 v_k29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_ex_k0 = v_k30
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k31,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
  end
 end
 v_k30 = v_ex_k0
 v_k31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
 end
 if (v_bit != 0)
  v__t = v_k16
  v_k16 = v_k17
  v_k17 = v__t
 end
 if (v_bit != 0)
  v__t = v_k18
  v_k18 = v_k19
  v_k19 = v__t
 end
 if (v_bit != 0)
  v__t = v_k20
  v_k20 = v_k21
  v_k21 = v__t
 end
 if (v_bit != 0)
  v__t = v_k22
  v_k22 = v_k23
  v_k23 = v__t
 end
 if (v_bit != 0)
  v__t = v_k24
  v_k24 = v_k25
  v_k25 = v__t
 end
 if (v_bit != 0)
  v__t = v_k26
  v_k26 = v_k27
  v_k27 = v__t
 end
 if (v_bit != 0)
  v__t = v_k28
  v_k28 = v_k29
  v_k29 = v__t
 end
 if (v_bit != 0)
  v__t = v_k30
  v_k30 = v_k31
  v_k31 = v__t
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_v0,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k0,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v0,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_v0 = v_ex_v0
 end
 return (v_k0,v_v0,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_k1,v_v0,v_v1,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 end
 return (v_k0,v_k1,v_v0,v_v1,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_k1,v_k2,v_k3,v_v0,v_v1,v_v2,v_v3,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k2
  v_k2 = v__t
  v__s = v_v0
  v_v0 = v_v2
  v_v2 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k3
  v_k3 = v__t
  v__s = v_v1
  v_v1 = v_v3
  v_v3 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k2
  v_k2 = v__t
  v__s = v_v0
  v_v0 = v_v2
  v_v2 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k3
  v_k3 = v__t
  v__s = v_v1
  v_v1 = v_v3
  v_v3 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_v0,v_v1,v_v2,v_v3,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k6
  v_k6 = v__t
  v__s = v_v0
  v_v0 = v_v6
  v_v6 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k7
  v_k7 = v__t
  v__s = v_v1
  v_v1 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k4
  v_k4 = v__t
  v__s = v_v2
  v_v2 = v_v4
  v_v4 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k5
  v_k5 = v__t
  v__s = v_v3
  v_v3 = v_v5
  v_v5 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k6
  v_k6 = v__t
  v__s = v_v0
  v_v0 = v_v6
  v_v6 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k7
  v_k7 = v__t
  v__s = v_v1
  v_v1 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k4
  v_k4 = v__t
  v__s = v_v2
  v_v2 = v_v4
  v_v4 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k5
  v_k5 = v__t
  v__s = v_v3
  v_v3 = v_v5
  v_v5 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k14
  v_k14 = v__t
  v__s = v_v0
  v_v0 = v_v14
  v_v14 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k15
  v_k15 = v__t
  v__s = v_v1
  v_v1 = v_v15
  v_v15 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k12
  v_k12 = v__t
  v__s = v_v2
  v_v2 = v_v12
  v_v12 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k13
  v_k13 = v__t
  v__s = v_v3
  v_v3 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k10
  v_k10 = v__t
  v__s = v_v4
  v_v4 = v_v10
  v_v10 = v__s
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k11
  v_k11 = v__t
  v__s = v_v5
  v_v5 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k8
  v_k8 = v__t
  v__s = v_v6
  v_v6 = v_v8
  v_v8 = v__s
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k9
  v_k9 = v__t
  v__s = v_v7
  v_v7 = v_v9
  v_v9 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 v_ex_v0 = v_v8
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v8 = v_ex_v0
 v_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 v_ex_v0 = v_v10
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v10 = v_ex_v0
 v_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 v_ex_v0 = v_v12
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v12 = v_ex_v0
 v_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 v_ex_v0 = v_v14
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v14 = v_ex_v0
 v_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k14
  v_k14 = v__t
  v__s = v_v0
  v_v0 = v_v14
  v_v14 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k15
  v_k15 = v__t
  v__s = v_v1
  v_v1 = v_v15
  v_v15 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k12
  v_k12 = v__t
  v__s = v_v2
  v_v2 = v_v12
  v_v12 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k13
  v_k13 = v__t
  v__s = v_v3
  v_v3 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k10
  v_k10 = v__t
  v__s = v_v4
  v_v4 = v_v10
  v_v10 = v__s
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k11
  v_k11 = v__t
  v__s = v_v5
  v_v5 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k8
  v_k8 = v__t
  v__s = v_v6
  v_v6 = v_v8
  v_v8 = v__s
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k9
  v_k9 = v__t
  v__s = v_v7
  v_v7 = v_v9
  v_v9 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_intxn(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_v16,v_v17,v_v18,v_v19,v_v20,v_v21,v_v22,v_v23,v_v24,v_v25,v_v26,v_v27,v_v28,v_v29,v_v30,v_v31,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k30
  v_k30 = v__t
  v__s = v_v0
  v_v0 = v_v30
  v_v30 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k31
  v_k31 = v__t
  v__s = v_v1
  v_v1 = v_v31
  v_v31 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k28
  v_k28 = v__t
  v__s = v_v2
  v_v2 = v_v28
  v_v28 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k29
  v_k29 = v__t
  v__s = v_v3
  v_v3 = v_v29
  v_v29 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k26
  v_k26 = v__t
  v__s = v_v4
  v_v4 = v_v26
  v_v26 = v__s
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k27
  v_k27 = v__t
  v__s = v_v5
  v_v5 = v_v27
  v_v27 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k24
  v_k24 = v__t
  v__s = v_v6
  v_v6 = v_v24
  v_v24 = v__s
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k25
  v_k25 = v__t
  v__s = v_v7
  v_v7 = v_v25
  v_v25 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k22
  v_k22 = v__t
  v__s = v_v8
  v_v8 = v_v22
  v_v22 = v__s
 end
 if (v_bit != 0)
  v__t = v_k9
  v_k9 = v_k23
  v_k23 = v__t
  v__s = v_v9
  v_v9 = v_v23
  v_v23 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k20
  v_k20 = v__t
  v__s = v_v10
  v_v10 = v_v20
  v_v20 = v__s
 end
 if (v_bit != 0)
  v__t = v_k11
  v_k11 = v_k21
  v_k21 = v__t
  v__s = v_v11
  v_v11 = v_v21
  v_v21 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k18
  v_k18 = v__t
  v__s = v_v12
  v_v12 = v_v18
  v_v18 = v__s
 end
 if (v_bit != 0)
  v__t = v_k13
  v_k13 = v_k19
  v_k19 = v__t
  v__s = v_v13
  v_v13 = v_v19
  v_v19 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k16
  v_k16 = v__t
  v__s = v_v14
  v_v14 = v_v16
  v_v16 = v__s
 end
 if (v_bit != 0)
  v__t = v_k15
  v_k15 = v_k17
  v_k17 = v__t
  v__s = v_v15
  v_v15 = v_v17
  v_v17 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 v_ex_v0 = v_v8
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v8 = v_ex_v0
 v_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 v_ex_v0 = v_v10
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v10 = v_ex_v0
 v_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 v_ex_v0 = v_v12
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v12 = v_ex_v0
 v_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 v_ex_v0 = v_v14
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v14 = v_ex_v0
 v_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k16
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k17,v_mask)
 v_ex_v0 = v_v16
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v17,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k16 = v_ex_k0
 v_k17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v16 = v_ex_v0
 v_v17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k18
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k19,v_mask)
 v_ex_v0 = v_v18
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v19,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k18 = v_ex_k0
 v_k19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v18 = v_ex_v0
 v_v19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k20
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k21,v_mask)
 v_ex_v0 = v_v20
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v21,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k20 = v_ex_k0
 v_k21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v20 = v_ex_v0
 v_v21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k22
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k23,v_mask)
 v_ex_v0 = v_v22
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v23,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k22 = v_ex_k0
 v_k23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v22 = v_ex_v0
 v_v23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k24
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k25,v_mask)
 v_ex_v0 = v_v24
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v25,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k24 = v_ex_k0
 v_k25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v24 = v_ex_v0
 v_v25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k26
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k27,v_mask)
 v_ex_v0 = v_v26
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v27,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k26 = v_ex_k0
 v_k27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v26 = v_ex_v0
 v_v27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k28
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k29,v_mask)
 v_ex_v0 = v_v28
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v29,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k28 = v_ex_k0
 v_k29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v28 = v_ex_v0
 v_v29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k30
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k31,v_mask)
 v_ex_v0 = v_v30
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v31,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k30 = v_ex_k0
 v_k31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v30 = v_ex_v0
 v_v31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k30
  v_k30 = v__t
  v__s = v_v0
  v_v0 = v_v30
  v_v30 = v__s
 end
 if (v_bit != 0)
  v__t = v_k1
  v_k1 = v_k31
  v_k31 = v__t
  v__s = v_v1
  v_v1 = v_v31
  v_v31 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k28
  v_k28 = v__t
  v__s = v_v2
  v_v2 = v_v28
  v_v28 = v__s
 end
 if (v_bit != 0)
  v__t = v_k3
  v_k3 = v_k29
  v_k29 = v__t
  v__s = v_v3
  v_v3 = v_v29
  v_v29 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k26
  v_k26 = v__t
  v__s = v_v4
  v_v4 = v_v26
  v_v26 = v__s
 end
 if (v_bit != 0)
  v__t = v_k5
  v_k5 = v_k27
  v_k27 = v__t
  v__s = v_v5
  v_v5 = v_v27
  v_v27 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k24
  v_k24 = v__t
  v__s = v_v6
  v_v6 = v_v24
  v_v24 = v__s
 end
 if (v_bit != 0)
  v__t = v_k7
  v_k7 = v_k25
  v_k25 = v__t
  v__s = v_v7
  v_v7 = v_v25
  v_v25 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k22
  v_k22 = v__t
  v__s = v_v8
  v_v8 = v_v22
  v_v22 = v__s
 end
 if (v_bit != 0)
  v__t = v_k9
  v_k9 = v_k23
  v_k23 = v__t
  v__s = v_v9
  v_v9 = v_v23
  v_v23 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k20
  v_k20 = v__t
  v__s = v_v10
  v_v10 = v_v20
  v_v20 = v__s
 end
 if (v_bit != 0)
  v__t = v_k11
  v_k11 = v_k21
  v_k21 = v__t
  v__s = v_v11
  v_v11 = v_v21
  v_v21 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k18
  v_k18 = v__t
  v__s = v_v12
  v_v12 = v_v18
  v_v18 = v__s
 end
 if (v_bit != 0)
  v__t = v_k13
  v_k13 = v_k19
  v_k19 = v__t
  v__s = v_v13
  v_v13 = v_v19
  v_v19 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k16
  v_k16 = v__t
  v__s = v_v14
  v_v14 = v_v16
  v_v16 = v__s
 end
 if (v_bit != 0)
  v__t = v_k15
  v_k15 = v_k17
  v_k17 = v__t
  v__s = v_v15
  v_v15 = v_v17
  v_v17 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_v16,v_v17,v_v18,v_v19,v_v20,v_v21,v_v22,v_v23,v_v24,v_v25,v_v26,v_v27,v_v28,v_v29,v_v30,v_v31,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_v0,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k0,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v0,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_v0 = v_ex_v0
 end
 return (v_k0,v_v0,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_k1,v_v0,v_v1,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 end
 return (v_k0,v_k1,v_v0,v_v1,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_k1,v_k2,v_k3,v_v0,v_v1,v_v2,v_v3,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_v0,v_v1,v_v2,v_v3,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
  v__s = v_v8
  v_v8 = v_v9
  v_v9 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
  v__s = v_v10
  v_v10 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
  v__s = v_v12
  v_v12 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
  v__s = v_v14
  v_v14 = v_v15
  v_v15 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 v_ex_v0 = v_v8
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v8 = v_ex_v0
 v_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 v_ex_v0 = v_v10
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v10 = v_ex_v0
 v_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 v_ex_v0 = v_v12
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v12 = v_ex_v0
 v_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 v_ex_v0 = v_v14
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v14 = v_ex_v0
 v_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
  v__s = v_v8
  v_v8 = v_v9
  v_v9 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
  v__s = v_v10
  v_v10 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
  v__s = v_v12
  v_v12 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
  v__s = v_v14
  v_v14 = v_v15
  v_v15 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,)
end

# bb_exch.cuh 0205ab6bf94843314a7b6f74b2ea7a7cef17ac997211c996e0bb1f2578446fab
@inline function sg_exch_paral(ctx,v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_v16,v_v17,v_v18,v_v19,v_v20,v_v21,v_v22,v_v23,v_v24,v_v25,v_v26,v_v27,v_v28,v_v29,v_v30,v_v31,v_mask,v_bit)
 @inbounds begin
 local v_ex_k0::Int32 = zero(Int32)
 local v_ex_k1::Int32 = zero(Int32)
 local v_ex_v0::Int32 = zero(Int32)
 local v_ex_v1::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
  v__s = v_v8
  v_v8 = v_v9
  v_v9 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
  v__s = v_v10
  v_v10 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
  v__s = v_v12
  v_v12 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
  v__s = v_v14
  v_v14 = v_v15
  v_v15 = v__s
 end
 if (v_bit != 0)
  v__t = v_k16
  v_k16 = v_k17
  v_k17 = v__t
  v__s = v_v16
  v_v16 = v_v17
  v_v17 = v__s
 end
 if (v_bit != 0)
  v__t = v_k18
  v_k18 = v_k19
  v_k19 = v__t
  v__s = v_v18
  v_v18 = v_v19
  v_v19 = v__s
 end
 if (v_bit != 0)
  v__t = v_k20
  v_k20 = v_k21
  v_k21 = v__t
  v__s = v_v20
  v_v20 = v_v21
  v_v21 = v__s
 end
 if (v_bit != 0)
  v__t = v_k22
  v_k22 = v_k23
  v_k23 = v__t
  v__s = v_v22
  v_v22 = v_v23
  v_v23 = v__s
 end
 if (v_bit != 0)
  v__t = v_k24
  v_k24 = v_k25
  v_k25 = v__t
  v__s = v_v24
  v_v24 = v_v25
  v_v25 = v__s
 end
 if (v_bit != 0)
  v__t = v_k26
  v_k26 = v_k27
  v_k27 = v__t
  v__s = v_v26
  v_v26 = v_v27
  v_v27 = v__s
 end
 if (v_bit != 0)
  v__t = v_k28
  v_k28 = v_k29
  v_k29 = v__t
  v__s = v_v28
  v_v28 = v_v29
  v_v29 = v__s
 end
 if (v_bit != 0)
  v__t = v_k30
  v_k30 = v_k31
  v_k31 = v__t
  v__s = v_v30
  v_v30 = v_v31
  v_v31 = v__s
 end
 v_ex_k0 = v_k0
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k1,v_mask)
 v_ex_v0 = v_v0
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v1,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k0 = v_ex_k0
 v_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v0 = v_ex_v0
 v_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k2
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k3,v_mask)
 v_ex_v0 = v_v2
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v3,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k2 = v_ex_k0
 v_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v2 = v_ex_v0
 v_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k4
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k5,v_mask)
 v_ex_v0 = v_v4
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v5,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k4 = v_ex_k0
 v_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v4 = v_ex_v0
 v_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k6
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k7,v_mask)
 v_ex_v0 = v_v6
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v7,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k6 = v_ex_k0
 v_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v6 = v_ex_v0
 v_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k8
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k9,v_mask)
 v_ex_v0 = v_v8
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v9,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k8 = v_ex_k0
 v_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v8 = v_ex_v0
 v_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k10
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k11,v_mask)
 v_ex_v0 = v_v10
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v11,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k10 = v_ex_k0
 v_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v10 = v_ex_v0
 v_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k12
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k13,v_mask)
 v_ex_v0 = v_v12
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v13,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k12 = v_ex_k0
 v_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v12 = v_ex_v0
 v_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k14
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k15,v_mask)
 v_ex_v0 = v_v14
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v15,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k14 = v_ex_k0
 v_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v14 = v_ex_v0
 v_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k16
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k17,v_mask)
 v_ex_v0 = v_v16
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v17,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k16 = v_ex_k0
 v_k17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v16 = v_ex_v0
 v_v17 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k18
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k19,v_mask)
 v_ex_v0 = v_v18
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v19,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k18 = v_ex_k0
 v_k19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v18 = v_ex_v0
 v_v19 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k20
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k21,v_mask)
 v_ex_v0 = v_v20
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v21,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k20 = v_ex_k0
 v_k21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v20 = v_ex_v0
 v_v21 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k22
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k23,v_mask)
 v_ex_v0 = v_v22
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v23,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k22 = v_ex_k0
 v_k23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v22 = v_ex_v0
 v_v23 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k24
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k25,v_mask)
 v_ex_v0 = v_v24
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v25,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k24 = v_ex_k0
 v_k25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v24 = v_ex_v0
 v_v25 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k26
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k27,v_mask)
 v_ex_v0 = v_v26
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v27,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k26 = v_ex_k0
 v_k27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v26 = v_ex_v0
 v_v27 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k28
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k29,v_mask)
 v_ex_v0 = v_v28
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v29,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k28 = v_ex_k0
 v_k29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v28 = v_ex_v0
 v_v29 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 v_ex_k0 = v_k30
 v_ex_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_k31,v_mask)
 v_ex_v0 = v_v30
 v_ex_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_v31,v_mask)
 if (v_ex_k0 > v_ex_k1)
  v__t = v_ex_k0
  v_ex_k0 = v_ex_k1
  v_ex_k1 = v__t
  v__s = v_ex_v0
  v_ex_v0 = v_ex_v1
  v_ex_v1 = v__s
 end
 if (v_bit != 0)
  if (v_ex_k0 != v_ex_k1)
   v__t = v_ex_k0
   v_ex_k0 = v_ex_k1
   v_ex_k1 = v__t
   v__s = v_ex_v0
   v_ex_v0 = v_ex_v1
   v_ex_v1 = v__s
  end
 end
 v_k30 = v_ex_k0
 v_k31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_k1,v_mask)
 v_v30 = v_ex_v0
 v_v31 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_ex_v1,v_mask)
 if (v_bit != 0)
  v__t = v_k0
  v_k0 = v_k1
  v_k1 = v__t
  v__s = v_v0
  v_v0 = v_v1
  v_v1 = v__s
 end
 if (v_bit != 0)
  v__t = v_k2
  v_k2 = v_k3
  v_k3 = v__t
  v__s = v_v2
  v_v2 = v_v3
  v_v3 = v__s
 end
 if (v_bit != 0)
  v__t = v_k4
  v_k4 = v_k5
  v_k5 = v__t
  v__s = v_v4
  v_v4 = v_v5
  v_v5 = v__s
 end
 if (v_bit != 0)
  v__t = v_k6
  v_k6 = v_k7
  v_k7 = v__t
  v__s = v_v6
  v_v6 = v_v7
  v_v7 = v__s
 end
 if (v_bit != 0)
  v__t = v_k8
  v_k8 = v_k9
  v_k9 = v__t
  v__s = v_v8
  v_v8 = v_v9
  v_v9 = v__s
 end
 if (v_bit != 0)
  v__t = v_k10
  v_k10 = v_k11
  v_k11 = v__t
  v__s = v_v10
  v_v10 = v_v11
  v_v11 = v__s
 end
 if (v_bit != 0)
  v__t = v_k12
  v_k12 = v_k13
  v_k13 = v__t
  v__s = v_v12
  v_v12 = v_v13
  v_v13 = v__s
 end
 if (v_bit != 0)
  v__t = v_k14
  v_k14 = v_k15
  v_k15 = v__t
  v__s = v_v14
  v_v14 = v_v15
  v_v15 = v__s
 end
 if (v_bit != 0)
  v__t = v_k16
  v_k16 = v_k17
  v_k17 = v__t
  v__s = v_v16
  v_v16 = v_v17
  v_v17 = v__s
 end
 if (v_bit != 0)
  v__t = v_k18
  v_k18 = v_k19
  v_k19 = v__t
  v__s = v_v18
  v_v18 = v_v19
  v_v19 = v__s
 end
 if (v_bit != 0)
  v__t = v_k20
  v_k20 = v_k21
  v_k21 = v__t
  v__s = v_v20
  v_v20 = v_v21
  v_v21 = v__s
 end
 if (v_bit != 0)
  v__t = v_k22
  v_k22 = v_k23
  v_k23 = v__t
  v__s = v_v22
  v_v22 = v_v23
  v_v23 = v__s
 end
 if (v_bit != 0)
  v__t = v_k24
  v_k24 = v_k25
  v_k25 = v__t
  v__s = v_v24
  v_v24 = v_v25
  v_v25 = v__s
 end
 if (v_bit != 0)
  v__t = v_k26
  v_k26 = v_k27
  v_k27 = v__t
  v__s = v_v26
  v_v26 = v_v27
  v_v27 = v__s
 end
 if (v_bit != 0)
  v__t = v_k28
  v_k28 = v_k29
  v_k29 = v__t
  v__s = v_v28
  v_v28 = v_v29
  v_v29 = v__s
 end
 if (v_bit != 0)
  v__t = v_k30
  v_k30 = v_k31
  v_k31 = v__t
  v__s = v_v30
  v_v30 = v_v31
  v_v31 = v__s
 end
 end
 return (v_k0,v_k1,v_k2,v_k3,v_k4,v_k5,v_k6,v_k7,v_k8,v_k9,v_k10,v_k11,v_k12,v_k13,v_k14,v_k15,v_k16,v_k17,v_k18,v_k19,v_k20,v_k21,v_k22,v_k23,v_k24,v_k25,v_k26,v_k27,v_k28,v_k29,v_k30,v_k31,v_v0,v_v1,v_v2,v_v3,v_v4,v_v5,v_v6,v_v7,v_v8,v_v9,v_v10,v_v11,v_v12,v_v13,v_v14,v_v15,v_v16,v_v17,v_v18,v_v19,v_v20,v_v21,v_v22,v_v23,v_v24,v_v25,v_v26,v_v27,v_v28,v_v29,v_v30,v_v31,)
end

# bb_bin.cuh 8573c488f38b2e0bbbc497fac3ca104075504bb23abcd67a19e43017c84ee895
@inline function sg_warp_exclusive_sum(ctx,v_in,v_out,v_n)
 @inbounds begin
 local v_lane::Int32 = zero(Int32)
 local v_data::Int32 = zero(Int32)
 local v_i::Int32 = zero(Int32)
 local v_other::Int32 = zero(Int32)
 v_lane = (ctx[1] & Int32(31))
 v_data = (((v_lane > Int32(0)) && (v_lane < v_n)) ? v_in[(v_lane - Int32(1))] : Int32(0))
 v_i = Int32(1)
 while (v_i < Int32(32))
  v_other = CUDA.shfl_up_sync(UInt32(0xFFFFFFFF),v_data,v_i)
  if (v_lane > v_i)
   v_data += v_other
  end
  v_i *= Int32(2)
 end
 if (v_lane < v_n)
  v_out[v_lane] = v_data
 end
 end
 nothing
end

# bb_bin.cuh 8573c488f38b2e0bbbc497fac3ca104075504bb23abcd67a19e43017c84ee895
@inline function sg_bb_bin_histo(ctx,v_d_bin_counter,v_d_seg_begins,v_d_seg_ends,v_num_segs)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_gid::Int32 = zero(Int32)
 v_local_histo=CSpan(CUDA.CuStaticSharedArray(Int32,(Int32(13) + Int32(1))),Int32(0))
 local v_size::Int32 = zero(Int32)
 v_tid = ctx[1]
 v_gid = ((ctx[2] * ctx[4]) + ctx[1])
 if (v_tid < (Int32(13) + Int32(1)))
  v_local_histo[v_tid] = Int32(0)
 end
 CUDA.sync_threads()
 if (v_gid < v_num_segs)
  v_size = (v_d_seg_ends[v_gid] - v_d_seg_begins[v_gid])
  if (v_size <= Int32(1))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(0)),Int32(1))
  end
  if ((Int32(1) < v_size) && (v_size <= Int32(2)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(1)),Int32(1))
  end
  if ((Int32(2) < v_size) && (v_size <= Int32(4)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(2)),Int32(1))
  end
  if ((Int32(4) < v_size) && (v_size <= Int32(8)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(3)),Int32(1))
  end
  if ((Int32(8) < v_size) && (v_size <= Int32(16)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(4)),Int32(1))
  end
  if ((Int32(16) < v_size) && (v_size <= Int32(32)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(5)),Int32(1))
  end
  if ((Int32(32) < v_size) && (v_size <= Int32(64)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(6)),Int32(1))
  end
  if ((Int32(64) < v_size) && (v_size <= Int32(128)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(7)),Int32(1))
  end
  if ((Int32(128) < v_size) && (v_size <= Int32(256)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(8)),Int32(1))
  end
  if ((Int32(256) < v_size) && (v_size <= Int32(512)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(9)),Int32(1))
  end
  if ((Int32(512) < v_size) && (v_size <= Int32(1024)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(10)),Int32(1))
  end
  if ((Int32(1024) < v_size) && (v_size <= Int32(2048)))
   CUDA.atomic_add!(sg_pointer(v_local_histo,Int32(11)),Int32(1))
  end
  if (Int32(2048) < v_size)
   CUDA.atomic_max!(sg_pointer(v_local_histo,Int32(13)),v_size)
  end
 end
 CUDA.sync_threads()
 if (v_tid < Int32(32))
  sg_warp_exclusive_sum(ctx ,v_local_histo,v_local_histo,Int32(13))
  if (v_tid < Int32(13))
   CUDA.atomic_add!(sg_pointer(v_d_bin_counter,v_tid),v_local_histo[v_tid])
  end
  if (v_tid == Int32(13))
   CUDA.atomic_max!(sg_pointer(v_d_bin_counter,v_tid),v_local_histo[v_tid])
  end
 end
 end
 nothing
end

# bb_bin.cuh 8573c488f38b2e0bbbc497fac3ca104075504bb23abcd67a19e43017c84ee895
@inline function sg_bb_bin_group(ctx,v_d_bin_segs_id,v_d_bin_counter,v_d_seg_begins,v_d_seg_ends,v_num_segs)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_size::Int32 = zero(Int32)
 local v_position::Int32 = zero(Int32)
 v_gid = ((ctx[2] * ctx[4]) + ctx[1])
 if (v_gid < v_num_segs)
  v_size = (v_d_seg_ends[v_gid] - v_d_seg_begins[v_gid])
  if (v_size <= Int32(1))
   v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(0)),Int32(1))
  else
   if (v_size <= Int32(2))
    v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(1)),Int32(1))
   else
    if (v_size <= Int32(4))
     v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(2)),Int32(1))
    else
     if (v_size <= Int32(8))
      v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(3)),Int32(1))
     else
      if (v_size <= Int32(16))
       v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(4)),Int32(1))
      else
       if (v_size <= Int32(32))
        v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(5)),Int32(1))
       else
        if (v_size <= Int32(64))
         v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(6)),Int32(1))
        else
         if (v_size <= Int32(128))
          v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(7)),Int32(1))
         else
          if (v_size <= Int32(256))
           v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(8)),Int32(1))
          else
           if (v_size <= Int32(512))
            v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(9)),Int32(1))
           else
            if (v_size <= Int32(1024))
             v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(10)),Int32(1))
            else
             if (v_size <= Int32(2048))
              v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(11)),Int32(1))
             else
              v_position = CUDA.atomic_add!(sg_pointer(v_d_bin_counter,Int32(12)),Int32(1))
             end
            end
           end
          end
         end
        end
       end
      end
     end
    end
   end
  end
  v_d_bin_segs_id[v_position] = v_gid
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_copy(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = v_gid
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  if (v_seg_size == Int32(1))
   v_keyB[v_k] = v_key[v_k]
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk256_wp2_tc1_r2_r2_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k0 , = sg_exch_intxn_keys(ctx ,v_rg_k0,Int32(0x1),v_bit1)
  if (((v_tid << Int32(0)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(0))) + Int32(0))] = v_rg_k0
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_wp2_tc2_r3_r4_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(2)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(2))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_wp2_tc4_r5_r8_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(2)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(2))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(4)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(4))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(6)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(6))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_wp4_tc4_r9_r16_strd(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 local v_base::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(2))
 v_tid = (ctx[1] & Int32(3))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(8)) * Int32(8))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(4)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(4))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(12)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(12))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_base = (div(v_lane_id,Int32(16)) * Int32(16))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(4)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(4)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(12)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(12)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(20)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(20)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k3
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(28)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(28)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k3
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_wp8_tc4_r17_r32_strd(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(3))
 v_tid = (ctx[1] & Int32(7))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(4)) * Int32(4))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(24)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(24))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k3
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_wp16_tc4_r33_r64_strd(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(4))
 v_tid = (ctx[1] & Int32(15))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(2)) * Int32(2))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(48)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(48))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k3
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk256_wp8_tc16_r65_r128_strd(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_rg_k8::Int32 = zero(Int32)
 local v_rg_k9::Int32 = zero(Int32)
 local v_rg_k10::Int32 = zero(Int32)
 local v_rg_k11::Int32 = zero(Int32)
 local v_rg_k12::Int32 = zero(Int32)
 local v_rg_k13::Int32 = zero(Int32)
 local v_rg_k14::Int32 = zero(Int32)
 local v_rg_k15::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(3))
 v_tid = (ctx[1] & Int32(7))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(4)) * Int32(4))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(24)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(24))] : Int32(2147483647))
  v_rg_k4 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k5 = (((v_tid + Int32(40)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(40))] : Int32(2147483647))
  v_rg_k6 = (((v_tid + Int32(48)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(48))] : Int32(2147483647))
  v_rg_k7 = (((v_tid + Int32(56)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(56))] : Int32(2147483647))
  v_rg_k8 = (((v_tid + Int32(64)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(64))] : Int32(2147483647))
  v_rg_k9 = (((v_tid + Int32(72)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(72))] : Int32(2147483647))
  v_rg_k10 = (((v_tid + Int32(80)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(80))] : Int32(2147483647))
  v_rg_k11 = (((v_tid + Int32(88)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(88))] : Int32(2147483647))
  v_rg_k12 = (((v_tid + Int32(96)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(96))] : Int32(2147483647))
  v_rg_k13 = (((v_tid + Int32(104)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(104))] : Int32(2147483647))
  v_rg_k14 = (((v_tid + Int32(112)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(112))] : Int32(2147483647))
  v_rg_k15 = (((v_tid + Int32(120)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(120))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k4 > v_rg_k7)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k5 > v_rg_k6)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k8 > v_rg_k11)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k9 > v_rg_k10)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k12 > v_rg_k15)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k13 > v_rg_k14)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k7)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k1 > v_rg_k6)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k2 > v_rg_k5)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k3 > v_rg_k4)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k8 > v_rg_k15)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k9 > v_rg_k14)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k10 > v_rg_k13)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k11 > v_rg_k12)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k15)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k1 > v_rg_k14)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k2 > v_rg_k13)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k3 > v_rg_k12)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k4 > v_rg_k11)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k5 > v_rg_k10)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k6 > v_rg_k9)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k7 > v_rg_k8)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k8
   v_rg_k8 = v__t
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k2
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k4
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k6
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k8
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k10
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k12
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k14
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k3
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k5
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k7
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k9
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k11
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k13
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k15
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(4)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(4)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(4)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(4)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(4)) + Int32(4)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(4))] = v_rg_k4
   end
   if (((v_tid << Int32(4)) + Int32(5)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(5))] = v_rg_k5
   end
   if (((v_tid << Int32(4)) + Int32(6)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(6))] = v_rg_k6
   end
   if (((v_tid << Int32(4)) + Int32(7)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(7))] = v_rg_k7
   end
   if (((v_tid << Int32(4)) + Int32(8)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(8))] = v_rg_k8
   end
   if (((v_tid << Int32(4)) + Int32(9)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(9))] = v_rg_k9
   end
   if (((v_tid << Int32(4)) + Int32(10)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(10))] = v_rg_k10
   end
   if (((v_tid << Int32(4)) + Int32(11)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(11))] = v_rg_k11
   end
   if (((v_tid << Int32(4)) + Int32(12)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(12))] = v_rg_k12
   end
   if (((v_tid << Int32(4)) + Int32(13)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(13))] = v_rg_k13
   end
   if (((v_tid << Int32(4)) + Int32(14)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(14))] = v_rg_k14
   end
   if (((v_tid << Int32(4)) + Int32(15)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(15))] = v_rg_k15
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk256_wp32_tc8_r129_r256_strd(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(5))
 v_tid = (ctx[1] & Int32(31))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(1)) * Int32(1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(64)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(64))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(96)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(96))] : Int32(2147483647))
  v_rg_k4 = (((v_tid + Int32(128)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(128))] : Int32(2147483647))
  v_rg_k5 = (((v_tid + Int32(160)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(160))] : Int32(2147483647))
  v_rg_k6 = (((v_tid + Int32(192)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(192))] : Int32(2147483647))
  v_rg_k7 = (((v_tid + Int32(224)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(224))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k4 > v_rg_k7)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k5 > v_rg_k6)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k7)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k1 > v_rg_k6)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k2 > v_rg_k5)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k3 > v_rg_k4)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1f),v_bit5)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x8),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x8))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x8))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x8))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k4
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k1
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k5
  end
  if ((v_lane_id + Int32(128)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(128))] = v_rg_k2
  end
  if ((v_lane_id + Int32(160)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(160))] = v_rg_k6
  end
  if ((v_lane_id + Int32(192)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(192))] = v_rg_k3
  end
  if ((v_lane_id + Int32(224)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(224))] = v_rg_k7
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(3)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(3)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(3)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(3)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(3)) + Int32(4)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(4))] = v_rg_k4
   end
   if (((v_tid << Int32(3)) + Int32(5)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(5))] = v_rg_k5
   end
   if (((v_tid << Int32(3)) + Int32(6)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(6))] = v_rg_k6
   end
   if (((v_tid << Int32(3)) + Int32(7)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(7))] = v_rg_k7
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk128_tc4_r257_r512_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(512)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk256_tc4_r513_r1024_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(1024)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(7)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(7)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(7)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s_keys.cuh d206e5dcfc840ae9e0e79155538bf7051ff5eef3fa78235b5727ebd8f7a5fa41
@inline function sg_gen_bk512_tc4_r1025_r2048_orig(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   v_rg_k0,v_rg_k1 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(7)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(7)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(7)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(4)) << Int32(4))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((((((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(14)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(15)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(15)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(15)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(15)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(8))
   v_gran += ((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(9))
   v_gran += (((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(10))
   v_gran += ((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(11))
   v_gran += (((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(12))
   v_gran += ((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(13))
   v_gran += (((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(14))
   v_gran += ((((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(15))
   v_gran += (((((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(14)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_copy(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = v_gid
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  if (v_seg_size == Int32(1))
   v_keyB[v_k] = v_key[v_k]
   v_valB[v_k] = v_val[v_k]
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk256_wp2_tc1_r2_r2_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  v_rg_k0,v_rg_v0 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_v0,Int32(0x1),v_bit1)
  if (((v_tid << Int32(0)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(0))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(0)) + Int32(0)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(0))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_wp2_tc2_r3_r4_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(2)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(2))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(2)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(2))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
  end
  if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
  end
  if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_wp2_tc4_r5_r8_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(1))
 v_tid = (ctx[1] & Int32(1))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(2)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(2))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(4)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(4))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(6)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(6))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(2)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(2))
  end
  if ((v_tid + Int32(4)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(4))
  end
  if ((v_tid + Int32(6)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(6))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
  end
  if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_wp4_tc4_r9_r16_strd(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 local v_base::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(2))
 v_tid = (ctx[1] & Int32(3))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(8)) * Int32(8))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(4)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(4))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(12)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(12))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(4)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(4))
  end
  if ((v_tid + Int32(8)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(8))
  end
  if ((v_tid + Int32(12)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(12))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_base = (div(v_lane_id,Int32(16)) * Int32(16))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(4)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(4)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(12)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(12)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(20)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(20)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(0)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k3
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(28)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(28)+Int32(1)))
  if (((v_lane_id >> Int32(4)) == Int32(1)) && ((v_lane_id - v_base) < v_ss))
   v_keyB[((v_kk + v_lane_id) - v_base)] = v_rg_k3
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(0)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v0)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(4)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(4)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(1)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v0)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(0)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v2)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(12)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(12)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(1)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v2)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(0)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v1)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(20)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(20)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(1)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v1)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(0)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v3)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(28)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(28)+Int32(1)))
  if ((((v_lane_id >> Int32(4)) == Int32(1)) & ((v_lane_id - v_base) < v_ss)) != 0)
   v_valB[((v_kk + v_lane_id) - v_base)] = v_val[(v_kk + v_rg_v3)]
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_wp8_tc4_r17_r32_strd(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(3))
 v_tid = (ctx[1] & Int32(7))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(4)) * Int32(4))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(24)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(24))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(8)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(8))
  end
  if ((v_tid + Int32(16)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(16))
  end
  if ((v_tid + Int32(24)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(24))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k3
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v0)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v2)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v1)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v3)]
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_wp16_tc4_r33_r64_strd(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(4))
 v_tid = (ctx[1] & Int32(15))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(2)) * Int32(2))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(48)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(48))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(16)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(16))
  end
  if ((v_tid + Int32(32)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(32))
  end
  if ((v_tid + Int32(48)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(48))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x4))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x4))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x4))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x4))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k2
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k3
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v0)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v2)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v1)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v3)]
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(2)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(2)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
   if (((v_tid << Int32(2)) + Int32(2)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
   end
   if (((v_tid << Int32(2)) + Int32(3)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk256_wp8_tc16_r65_r128_strd(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_rg_k8::Int32 = zero(Int32)
 local v_rg_k9::Int32 = zero(Int32)
 local v_rg_k10::Int32 = zero(Int32)
 local v_rg_k11::Int32 = zero(Int32)
 local v_rg_k12::Int32 = zero(Int32)
 local v_rg_k13::Int32 = zero(Int32)
 local v_rg_k14::Int32 = zero(Int32)
 local v_rg_k15::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_rg_v4::Int32 = zero(Int32)
 local v_rg_v5::Int32 = zero(Int32)
 local v_rg_v6::Int32 = zero(Int32)
 local v_rg_v7::Int32 = zero(Int32)
 local v_rg_v8::Int32 = zero(Int32)
 local v_rg_v9::Int32 = zero(Int32)
 local v_rg_v10::Int32 = zero(Int32)
 local v_rg_v11::Int32 = zero(Int32)
 local v_rg_v12::Int32 = zero(Int32)
 local v_rg_v13::Int32 = zero(Int32)
 local v_rg_v14::Int32 = zero(Int32)
 local v_rg_v15::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(3))
 v_tid = (ctx[1] & Int32(7))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(4)) * Int32(4))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(8)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(8))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(16)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(16))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(24)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(24))] : Int32(2147483647))
  v_rg_k4 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k5 = (((v_tid + Int32(40)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(40))] : Int32(2147483647))
  v_rg_k6 = (((v_tid + Int32(48)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(48))] : Int32(2147483647))
  v_rg_k7 = (((v_tid + Int32(56)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(56))] : Int32(2147483647))
  v_rg_k8 = (((v_tid + Int32(64)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(64))] : Int32(2147483647))
  v_rg_k9 = (((v_tid + Int32(72)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(72))] : Int32(2147483647))
  v_rg_k10 = (((v_tid + Int32(80)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(80))] : Int32(2147483647))
  v_rg_k11 = (((v_tid + Int32(88)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(88))] : Int32(2147483647))
  v_rg_k12 = (((v_tid + Int32(96)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(96))] : Int32(2147483647))
  v_rg_k13 = (((v_tid + Int32(104)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(104))] : Int32(2147483647))
  v_rg_k14 = (((v_tid + Int32(112)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(112))] : Int32(2147483647))
  v_rg_k15 = (((v_tid + Int32(120)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(120))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(8)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(8))
  end
  if ((v_tid + Int32(16)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(16))
  end
  if ((v_tid + Int32(24)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(24))
  end
  if ((v_tid + Int32(32)) < v_seg_size)
   v_rg_v4 = (v_tid + Int32(32))
  end
  if ((v_tid + Int32(40)) < v_seg_size)
   v_rg_v5 = (v_tid + Int32(40))
  end
  if ((v_tid + Int32(48)) < v_seg_size)
   v_rg_v6 = (v_tid + Int32(48))
  end
  if ((v_tid + Int32(56)) < v_seg_size)
   v_rg_v7 = (v_tid + Int32(56))
  end
  if ((v_tid + Int32(64)) < v_seg_size)
   v_rg_v8 = (v_tid + Int32(64))
  end
  if ((v_tid + Int32(72)) < v_seg_size)
   v_rg_v9 = (v_tid + Int32(72))
  end
  if ((v_tid + Int32(80)) < v_seg_size)
   v_rg_v10 = (v_tid + Int32(80))
  end
  if ((v_tid + Int32(88)) < v_seg_size)
   v_rg_v11 = (v_tid + Int32(88))
  end
  if ((v_tid + Int32(96)) < v_seg_size)
   v_rg_v12 = (v_tid + Int32(96))
  end
  if ((v_tid + Int32(104)) < v_seg_size)
   v_rg_v13 = (v_tid + Int32(104))
  end
  if ((v_tid + Int32(112)) < v_seg_size)
   v_rg_v14 = (v_tid + Int32(112))
  end
  if ((v_tid + Int32(120)) < v_seg_size)
   v_rg_v15 = (v_tid + Int32(120))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k4 > v_rg_k7)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k5 > v_rg_k6)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k8 > v_rg_k11)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k9 > v_rg_k10)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k12 > v_rg_k15)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k13 > v_rg_k14)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k7)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k1 > v_rg_k6)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k2 > v_rg_k5)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k3 > v_rg_k4)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k8 > v_rg_k15)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k9 > v_rg_k14)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k10 > v_rg_k13)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k11 > v_rg_k12)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k15)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k1 > v_rg_k14)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k2 > v_rg_k13)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k3 > v_rg_k12)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k4 > v_rg_k11)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k5 > v_rg_k10)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k6 > v_rg_k9)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k7 > v_rg_k8)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v8
   v_rg_v8 = v__s
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v8
   v_rg_v8 = v__s
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v8
   v_rg_v8 = v__s
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_k8,v_rg_k9,v_rg_k10,v_rg_k11,v_rg_k12,v_rg_k13,v_rg_k14,v_rg_k15,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,v_rg_v8,v_rg_v9,v_rg_v10,v_rg_v11,v_rg_v12,v_rg_v13,v_rg_v14,v_rg_v15,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k8)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v8
   v_rg_v8 = v__s
  end
  if (v_rg_k1 > v_rg_k9)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k2 > v_rg_k10)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k3 > v_rg_k11)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k4 > v_rg_k12)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k5 > v_rg_k13)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k6 > v_rg_k14)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k7 > v_rg_k15)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k12)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if (v_rg_k9 > v_rg_k13)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k10 > v_rg_k14)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k11 > v_rg_k15)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k10)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if (v_rg_k9 > v_rg_k11)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k14)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if (v_rg_k13 > v_rg_k15)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k8 > v_rg_k9)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if (v_rg_k10 > v_rg_k11)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if (v_rg_k12 > v_rg_k13)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if (v_rg_k14 > v_rg_k15)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x1))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x1))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x1))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x1))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x1))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x1))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x2))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x2))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x2))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x2))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x2))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x2))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x4))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x4))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x4))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x4))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x4))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x4))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x4))
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_v8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v8,Int32(0x8))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x8))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x8))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x8))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x8))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x8))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x8))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v8
   v_rg_v8 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v10
   v_rg_v10 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v12
   v_rg_v12 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_v8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v8,Int32(0x8))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x8))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x8))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x8))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x8))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x8))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x8))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x10))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x10))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x10))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x10))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x10))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x10))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k2
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k4
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k6
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k8
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k10
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k12
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k14
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k1
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k3
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k5
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k7
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k9
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k11
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k13
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k15
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v0)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v2)]
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(64))] = v_val[(v_kk + v_rg_v4)]
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(96))] = v_val[(v_kk + v_rg_v6)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(8)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(8)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v8)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v10)]
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(64))] = v_val[(v_kk + v_rg_v12)]
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(96))] = v_val[(v_kk + v_rg_v14)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(16)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(16)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v1)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v3)]
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(64))] = v_val[(v_kk + v_rg_v5)]
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(96))] = v_val[(v_kk + v_rg_v7)]
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(24)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(24)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v9)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v11)]
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(64))] = v_val[(v_kk + v_rg_v13)]
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(96))] = v_val[(v_kk + v_rg_v15)]
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(4)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(4)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(4)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(4)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(4)) + Int32(4)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(4))] = v_rg_k4
   end
   if (((v_tid << Int32(4)) + Int32(5)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(5))] = v_rg_k5
   end
   if (((v_tid << Int32(4)) + Int32(6)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(6))] = v_rg_k6
   end
   if (((v_tid << Int32(4)) + Int32(7)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(7))] = v_rg_k7
   end
   if (((v_tid << Int32(4)) + Int32(8)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(8))] = v_rg_k8
   end
   if (((v_tid << Int32(4)) + Int32(9)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(9))] = v_rg_k9
   end
   if (((v_tid << Int32(4)) + Int32(10)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(10))] = v_rg_k10
   end
   if (((v_tid << Int32(4)) + Int32(11)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(11))] = v_rg_k11
   end
   if (((v_tid << Int32(4)) + Int32(12)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(12))] = v_rg_k12
   end
   if (((v_tid << Int32(4)) + Int32(13)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(13))] = v_rg_k13
   end
   if (((v_tid << Int32(4)) + Int32(14)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(14))] = v_rg_k14
   end
   if (((v_tid << Int32(4)) + Int32(15)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(4))) + Int32(15))] = v_rg_k15
   end
   if (((v_tid << Int32(4)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(4)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
   if (((v_tid << Int32(4)) + Int32(2)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
   end
   if (((v_tid << Int32(4)) + Int32(3)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
   end
   if (((v_tid << Int32(4)) + Int32(4)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(4))] = v_val[(v_k + v_rg_v4)]
   end
   if (((v_tid << Int32(4)) + Int32(5)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(5))] = v_val[(v_k + v_rg_v5)]
   end
   if (((v_tid << Int32(4)) + Int32(6)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(6))] = v_val[(v_k + v_rg_v6)]
   end
   if (((v_tid << Int32(4)) + Int32(7)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(7))] = v_val[(v_k + v_rg_v7)]
   end
   if (((v_tid << Int32(4)) + Int32(8)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(8))] = v_val[(v_k + v_rg_v8)]
   end
   if (((v_tid << Int32(4)) + Int32(9)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(9))] = v_val[(v_k + v_rg_v9)]
   end
   if (((v_tid << Int32(4)) + Int32(10)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(10))] = v_val[(v_k + v_rg_v10)]
   end
   if (((v_tid << Int32(4)) + Int32(11)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(11))] = v_val[(v_k + v_rg_v11)]
   end
   if (((v_tid << Int32(4)) + Int32(12)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(12))] = v_val[(v_k + v_rg_v12)]
   end
   if (((v_tid << Int32(4)) + Int32(13)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(13))] = v_val[(v_k + v_rg_v13)]
   end
   if (((v_tid << Int32(4)) + Int32(14)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(14))] = v_val[(v_k + v_rg_v14)]
   end
   if (((v_tid << Int32(4)) + Int32(15)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(4))) + Int32(15))] = v_val[(v_k + v_rg_v15)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk256_wp32_tc8_r129_r256_strd(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_gid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_rg_v4::Int32 = zero(Int32)
 local v_rg_v5::Int32 = zero(Int32)
 local v_rg_v6::Int32 = zero(Int32)
 local v_rg_v7::Int32 = zero(Int32)
 local v_normalized_bin_size::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v_kk::Int32 = zero(Int32)
 local v_ss::Int32 = zero(Int32)
 v_gid = (ctx[1] + (ctx[2] * ctx[4]))
 v_bin_it = (v_gid >> Int32(5))
 v_tid = (ctx[1] & Int32(31))
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_normalized_bin_size = (div(v_bin_size,Int32(1)) * Int32(1))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_rg_k0 = (((v_tid + Int32(0)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = (((v_tid + Int32(32)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(32))] : Int32(2147483647))
  v_rg_k2 = (((v_tid + Int32(64)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(64))] : Int32(2147483647))
  v_rg_k3 = (((v_tid + Int32(96)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(96))] : Int32(2147483647))
  v_rg_k4 = (((v_tid + Int32(128)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(128))] : Int32(2147483647))
  v_rg_k5 = (((v_tid + Int32(160)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(160))] : Int32(2147483647))
  v_rg_k6 = (((v_tid + Int32(192)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(192))] : Int32(2147483647))
  v_rg_k7 = (((v_tid + Int32(224)) < v_seg_size) ? v_key[((v_k + v_tid) + Int32(224))] : Int32(2147483647))
  if ((v_tid + Int32(0)) < v_seg_size)
   v_rg_v0 = (v_tid + Int32(0))
  end
  if ((v_tid + Int32(32)) < v_seg_size)
   v_rg_v1 = (v_tid + Int32(32))
  end
  if ((v_tid + Int32(64)) < v_seg_size)
   v_rg_v2 = (v_tid + Int32(64))
  end
  if ((v_tid + Int32(96)) < v_seg_size)
   v_rg_v3 = (v_tid + Int32(96))
  end
  if ((v_tid + Int32(128)) < v_seg_size)
   v_rg_v4 = (v_tid + Int32(128))
  end
  if ((v_tid + Int32(160)) < v_seg_size)
   v_rg_v5 = (v_tid + Int32(160))
  end
  if ((v_tid + Int32(192)) < v_seg_size)
   v_rg_v6 = (v_tid + Int32(192))
  end
  if ((v_tid + Int32(224)) < v_seg_size)
   v_rg_v7 = (v_tid + Int32(224))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k4 > v_rg_k7)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k5 > v_rg_k6)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k7)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k1 > v_rg_k6)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k2 > v_rg_k5)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k3 > v_rg_k4)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1f),v_bit5)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x8),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_k4,v_rg_k5,v_rg_k6,v_rg_k7,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,v_rg_v4,v_rg_v5,v_rg_v6,v_rg_v7,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k4)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if (v_rg_k1 > v_rg_k5)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k2 > v_rg_k6)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k3 > v_rg_k7)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k6)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if (v_rg_k5 > v_rg_k7)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k4 > v_rg_k5)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if (v_rg_k6 > v_rg_k7)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
 end
 if (v_bin_it < v_normalized_bin_size)
  v_lane_id = (ctx[1] & Int32(31))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x8))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x8))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x8))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x8))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x8))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x8))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x8))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x8))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x8))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x8))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x8))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(0))] = v_rg_k0
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(32))] = v_rg_k4
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(64))] = v_rg_k1
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(96))] = v_rg_k5
  end
  if ((v_lane_id + Int32(128)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(128))] = v_rg_k2
  end
  if ((v_lane_id + Int32(160)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(160))] = v_rg_k6
  end
  if ((v_lane_id + Int32(192)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(192))] = v_rg_k3
  end
  if ((v_lane_id + Int32(224)) < v_ss)
   v_keyB[((v_kk + v_lane_id) + Int32(224))] = v_rg_k7
  end
  v_kk = CUDA.shfl_sync(UInt32(0xffffffff),v_k,(Int32(0)+Int32(1)))
  v_ss = CUDA.shfl_sync(UInt32(0xffffffff),v_seg_size,(Int32(0)+Int32(1)))
  if ((v_lane_id + Int32(0)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(0))] = v_val[(v_kk + v_rg_v0)]
  end
  if ((v_lane_id + Int32(32)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(32))] = v_val[(v_kk + v_rg_v4)]
  end
  if ((v_lane_id + Int32(64)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(64))] = v_val[(v_kk + v_rg_v1)]
  end
  if ((v_lane_id + Int32(96)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(96))] = v_val[(v_kk + v_rg_v5)]
  end
  if ((v_lane_id + Int32(128)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(128))] = v_val[(v_kk + v_rg_v2)]
  end
  if ((v_lane_id + Int32(160)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(160))] = v_val[(v_kk + v_rg_v6)]
  end
  if ((v_lane_id + Int32(192)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(192))] = v_val[(v_kk + v_rg_v3)]
  end
  if ((v_lane_id + Int32(224)) < v_ss)
   v_valB[((v_kk + v_lane_id) + Int32(224))] = v_val[(v_kk + v_rg_v7)]
  end
 else
  if (v_bin_it < v_bin_size)
   if (((v_tid << Int32(3)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(3)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(3)) + Int32(2)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(2))] = v_rg_k2
   end
   if (((v_tid << Int32(3)) + Int32(3)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(3))] = v_rg_k3
   end
   if (((v_tid << Int32(3)) + Int32(4)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(4))] = v_rg_k4
   end
   if (((v_tid << Int32(3)) + Int32(5)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(5))] = v_rg_k5
   end
   if (((v_tid << Int32(3)) + Int32(6)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(6))] = v_rg_k6
   end
   if (((v_tid << Int32(3)) + Int32(7)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(3))) + Int32(7))] = v_rg_k7
   end
   if (((v_tid << Int32(3)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(3)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
   if (((v_tid << Int32(3)) + Int32(2)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(2))] = v_val[(v_k + v_rg_v2)]
   end
   if (((v_tid << Int32(3)) + Int32(3)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(3))] = v_val[(v_k + v_rg_v3)]
   end
   if (((v_tid << Int32(3)) + Int32(4)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(4))] = v_val[(v_k + v_rg_v4)]
   end
   if (((v_tid << Int32(3)) + Int32(5)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(5))] = v_val[(v_k + v_rg_v5)]
   end
   if (((v_tid << Int32(3)) + Int32(6)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(6))] = v_val[(v_k + v_rg_v6)]
   end
   if (((v_tid << Int32(3)) + Int32(7)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(3))) + Int32(7))] = v_val[(v_k + v_rg_v7)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk128_tc4_r257_r512_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(512)),Int32(0))
 v_tmem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(512)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_tmp_v0::Int32 = zero(Int32)
 local v_tmp_v1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   v_rg_v0 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(0))
   v_rg_v1 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(32))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size)
    v_rg_v0 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size)
    v_rg_v1 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size)
    v_rg_v2 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size)
    v_rg_v3 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_v1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_v1
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_v2
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_val[(v_k + v_rg_v0)]
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_val[(v_k + v_rg_v1)]
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_val[(v_k + v_rg_v2)]
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk256_tc4_r513_r1024_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(1024)),Int32(0))
 v_tmem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(1024)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_tmp_v0::Int32 = zero(Int32)
 local v_tmp_v1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   v_rg_v0 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(0))
   v_rg_v1 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(32))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size)
    v_rg_v0 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size)
    v_rg_v1 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size)
    v_rg_v2 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size)
    v_rg_v3 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_v1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_v1
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_v2
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(7)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(7)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(7)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_val[(v_k + v_rg_v0)]
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_val[(v_k + v_rg_v1)]
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_val[(v_k + v_rg_v2)]
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_s.cuh 48ba6585542a66a5f3cf1494c16d7b3aee10f0ddf600eadf37508143c1ddedf1
@inline function sg_gen_bk512_tc4_r1025_r2048_orig(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_bin_size)
 @inbounds begin
 local v_tid::Int32 = zero(Int32)
 local v_bin_it::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 v_tmem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_tid1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_ext_seg_size::Int32 = zero(Int32)
 local v_big_wp::Int32 = zero(Int32)
 local v_sml_wp::Int32 = zero(Int32)
 local v_sml_len::Int32 = zero(Int32)
 local v_big_warp_id::Int32 = zero(Int32)
 local v_sml_warp::Bool = zero(Bool)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_tmp_wp_id::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_tmp_v0::Int32 = zero(Int32)
 local v_tmp_v1::Int32 = zero(Int32)
 local v_start
 v_tid = ctx[1]
 v_bin_it = ctx[2]
 v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
 v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
 v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
 v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
 v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
 v_tid1 = (ctx[1] & Int32(31))
 v_warp_id = div(ctx[1],Int32(32))
 if (v_bin_it < v_bin_size)
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
  v_ext_seg_size = (div((v_seg_size + Int32(63)),Int32(64)) * Int32(64))
  v_big_wp = div((v_ext_seg_size - (ctx[4] * Int32(2))),Int32(64))
  v_sml_wp = (div(ctx[4],Int32(32)) - v_big_wp)
  v_sml_len = (v_sml_wp * Int32(64))
  v_big_warp_id = (((v_warp_id - v_sml_wp) < Int32(0)) ? Int32(0) : (v_warp_id - v_sml_wp))
  v_sml_warp = (v_warp_id < v_sml_wp)
  if (v_sml_warp != 0)
   v_rg_k0 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(0))]
   v_rg_k1 = v_key[(((v_k + (v_warp_id << Int32(6))) + v_tid1) + Int32(32))]
   v_rg_v0 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(0))
   v_rg_v1 = (((v_warp_id << Int32(6)) + v_tid1) + Int32(32))
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_v0,v_rg_v1,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
  else
   v_rg_k0 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
   v_rg_k1 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
   v_rg_k2 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
   v_rg_k3 = (((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size) ? v_key[((((v_k + v_sml_len) + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0)) < v_seg_size)
    v_rg_v0 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(0))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32)) < v_seg_size)
    v_rg_v1 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(32))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64)) < v_seg_size)
    v_rg_v2 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(64))
   end
   if ((((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96)) < v_seg_size)
    v_rg_v3 = (((v_sml_len + v_tid1) + (v_big_warp_id << Int32(7))) + Int32(96))
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k3)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k1 > v_rg_k2)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0xf),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1f),v_bit5)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x8),v_bit4)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
   v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
   if (v_rg_k0 > v_rg_k2)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k2
    v_rg_k2 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v2
    v_rg_v2 = v__s
   end
   if (v_rg_k1 > v_rg_k3)
    v__t = v_rg_k1
    v_rg_k1 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v1
    v_rg_v1 = v_rg_v3
    v_rg_v3 = v__s
   end
   if (v_rg_k0 > v_rg_k1)
    v__t = v_rg_k0
    v_rg_k0 = v_rg_k1
    v_rg_k1 = v__t
    v__s = v_rg_v0
    v_rg_v0 = v_rg_v1
    v_rg_v1 = v__s
   end
   if (v_rg_k2 > v_rg_k3)
    v__t = v_rg_k2
    v_rg_k2 = v_rg_k3
    v_rg_k3 = v__t
    v__s = v_rg_v2
    v_rg_v2 = v_rg_v3
    v_rg_v3 = v__s
   end
  end
  if (v_sml_warp != 0)
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_k0
   v_smem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_k1
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_warp_id << Int32(6)) + (v_tid1 << Int32(1))) + Int32(1))] = v_rg_v1
  else
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_k0
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_k1
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_k2
   v_smem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_k3
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(0))] = v_rg_v0
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(1))] = v_rg_v1
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(2))] = v_rg_v2
   v_tmem[(((v_sml_len + (v_big_warp_id << Int32(7))) + (v_tid1 << Int32(2))) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_rhs_len = (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(3)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(3)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(3)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(3)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(7)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(7)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(7)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(7)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  CUDA.sync_threads()
  if (v_sml_warp != 0)
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  else
   v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
   v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
   v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
   v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
   v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
   v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
   v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
   v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  end
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(4)) << Int32(4))
  v_grp_start_off = ((v_grp_start_wp_id < v_sml_wp) ? (v_grp_start_wp_id * Int32(64)) : (v_sml_len + ((v_grp_start_wp_id - v_sml_wp) * Int32(128))))
  v_tmp_wp_id = v_grp_start_wp_id
  v_lhs_len = ((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_rhs_len = ((((((((((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(14)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(15)) < v_sml_wp) ? Int32(64) : Int32(128)))
  v_gran = ((v_warp_id < v_sml_wp) ? (v_tid1 << Int32(1)) : (v_tid1 << Int32(2)))
  if ((v_warp_id & Int32(15)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(15)) == Int32(1))
   v_gran += (((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128))
  end
  if ((v_warp_id & Int32(15)) == Int32(2))
   v_gran += ((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(3))
   v_gran += (((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(4))
   v_gran += ((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(5))
   v_gran += (((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(6))
   v_gran += ((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(7))
   v_gran += (((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(8))
   v_gran += ((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(9))
   v_gran += (((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(10))
   v_gran += ((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(11))
   v_gran += (((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(12))
   v_gran += ((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(13))
   v_gran += (((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(14))
   v_gran += ((((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  if ((v_warp_id & Int32(15)) == Int32(15))
   v_gran += (((((((((((((((((v_tmp_wp_id + Int32(0)) < v_sml_wp) ? Int32(64) : Int32(128)) + (((v_tmp_wp_id + Int32(1)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(2)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(3)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(4)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(5)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(6)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(7)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(8)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(9)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(10)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(11)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(12)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(13)) < v_sml_wp) ? Int32(64) : Int32(128))) + (((v_tmp_wp_id + Int32(14)) < v_sml_wp) ? Int32(64) : Int32(128)))
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  if (v_sml_warp != 0)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  else
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
   if (v_p != 0)
    v_s_a += Int32(1)
    v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
    if (v_s_a < v_lhs_len)
     v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
    end
   else
    v_s_b += Int32(1)
    v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
    if (v_s_b < (v_lhs_len + v_rhs_len))
     v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
    end
   end
   v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
   v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
   v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  end
  if (v_sml_warp != 0)
  else
  end
  if (v_sml_warp != 0)
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_rg_k0
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_keyB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_rg_k1
   end
   if (((v_tid << Int32(1)) + Int32(0)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(0))] = v_val[(v_k + v_rg_v0)]
   end
   if (((v_tid << Int32(1)) + Int32(1)) < v_seg_size)
    v_valB[((v_k + (v_tid << Int32(1))) + Int32(1))] = v_val[(v_k + v_rg_v1)]
   end
  else
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_rg_k0
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_rg_k1
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_rg_k2
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_keyB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_rg_k3
   end
   if ((((v_tid << Int32(2)) + Int32(0)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(0)) - v_sml_len)] = v_val[(v_k + v_rg_v0)]
   end
   if ((((v_tid << Int32(2)) + Int32(1)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(1)) - v_sml_len)] = v_val[(v_k + v_rg_v1)]
   end
   if ((((v_tid << Int32(2)) + Int32(2)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(2)) - v_sml_len)] = v_val[(v_k + v_rg_v2)]
   end
   if ((((v_tid << Int32(2)) + Int32(3)) - v_sml_len) < v_seg_size)
    v_valB[(((v_k + (v_tid << Int32(2))) + Int32(3)) - v_sml_len)] = v_val[(v_k + v_rg_v3)]
   end
  end
 end
 end
 nothing
end

# bb_comput_l_keys.cuh 9ed7f8a61b4249b6254de2d898bec752b1f0126c27672b795bfe938f9d5abc2f
@inline function sg_kern_block_sort(ctx,v_key,v_keyB,v_seg_begins,v_seg_ends,v_bin,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_warp_lane::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_inner_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_start
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if (v_innerbid < v_blk_stat)
  v_tid = ctx[1]
  v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
  v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
  v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
  v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
  v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
  v_warp_lane = (ctx[1] & Int32(31))
  v_warp_id = div(ctx[1],Int32(32))
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_k = (v_k + (v_innerbid << Int32(11)))
  v_inner_seg_size = min((v_seg_size - (v_innerbid << Int32(11))),Int32(2048))
  v_rg_k0 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(0)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(32)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
  v_rg_k2 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(64)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
  v_rg_k3 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(96)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_intxn_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1f),v_bit5)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x8),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3 = sg_exch_paral_keys(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(0))] = v_rg_k0
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(1))] = v_rg_k1
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(2))] = v_rg_k2
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(3))] = v_rg_k3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(128)
  v_rhs_len = Int32(128)
  v_gran = (v_warp_lane << Int32(2))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += Int32(128)
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(256)
  v_rhs_len = Int32(256)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(3)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(512)
  v_rhs_len = Int32(512)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(7)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(4)) << Int32(4))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(1024)
  v_rhs_len = Int32(1024)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(15)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (((v_tid << Int32(2)) + Int32(0)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
  end
 end
 end
 nothing
end

# bb_comput_l_keys.cuh 9ed7f8a61b4249b6254de2d898bec752b1f0126c27672b795bfe938f9d5abc2f
@inline function sg_kern_block_merge(ctx,v_keys,v_keysB,v_seg_begins,v_seg_ends,v_bin,v_stride,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,(Int32(128) * Int32(16))),Int32(0))
 local v_k::Int32 = zero(Int32)
 local v_loc_a::Int32 = zero(Int32)
 local v_loc_b::Int32 = zero(Int32)
 local v_cnt_a::Int32 = zero(Int32)
 local v_cnt_b::Int32 = zero(Int32)
 local v_coop::Int32 = zero(Int32)
 local v_coop_bid::Int32 = zero(Int32)
 local v_l_gran::Int32 = zero(Int32)
 local v_r_gran::Int32 = zero(Int32)
 local v_l_s_a::Int32 = zero(Int32)
 local v_l_s_b::Int32 = zero(Int32)
 local v_r_s_a::Int32 = zero(Int32)
 local v_r_s_b::Int32 = zero(Int32)
 local v_l_st::Int32 = zero(Int32)
 local v_l_cnt::Int32 = zero(Int32)
 local v_r_st::Int32 = zero(Int32)
 local v_r_cnt::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_rg_k8::Int32 = zero(Int32)
 local v_rg_k9::Int32 = zero(Int32)
 local v_rg_k10::Int32 = zero(Int32)
 local v_rg_k11::Int32 = zero(Int32)
 local v_rg_k12::Int32 = zero(Int32)
 local v_rg_k13::Int32 = zero(Int32)
 local v_rg_k14::Int32 = zero(Int32)
 local v_rg_k15::Int32 = zero(Int32)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if ((v_innerbid < v_blk_stat) && (v_stride < v_seg_size))
  v_tid = ctx[1]
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_coop = ((v_stride << Int32(1)) >> Int32(11))
  v_coop_bid = (v_innerbid % v_coop)
  v_loc_a = (div(v_innerbid,v_coop) * (v_stride << Int32(1)))
  v_cnt_a = min(v_stride,(v_seg_size - v_loc_a))
  v_loc_b = min((v_loc_a + v_stride),v_seg_size)
  v_cnt_b = min(v_stride,(v_seg_size - v_loc_b))
  v_l_gran = (v_coop_bid << Int32(11))
  v_r_gran = min(((v_coop_bid + Int32(1)) << Int32(11)),(v_seg_size - v_loc_a))
  v_l_s_a = sg_find_kth3(ctx ,((v_keys + v_k) + v_loc_a),v_cnt_a,((v_keys + v_k) + v_loc_b),v_cnt_b,v_l_gran)
  v_l_s_b = (v_l_gran - v_l_s_a)
  v_r_s_a = sg_find_kth3(ctx ,((v_keys + v_k) + v_loc_a),v_cnt_a,((v_keys + v_k) + v_loc_b),v_cnt_b,v_r_gran)
  v_r_s_b = (v_r_gran - v_r_s_a)
  v_l_st = Int32(0)
  v_l_cnt = (v_r_s_a - v_l_s_a)
  if ((v_l_s_a + v_tid) < v_r_s_a)
   v_smem[(v_l_st + v_tid)] = v_keys[(((v_k + v_loc_a) + v_l_s_a) + v_tid)]
  end
  if (((v_l_s_a + v_tid) + Int32(128)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(128))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(128))]
  end
  if (((v_l_s_a + v_tid) + Int32(256)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(256))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(256))]
  end
  if (((v_l_s_a + v_tid) + Int32(384)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(384))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(384))]
  end
  if (((v_l_s_a + v_tid) + Int32(512)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(512))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(512))]
  end
  if (((v_l_s_a + v_tid) + Int32(640)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(640))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(640))]
  end
  if (((v_l_s_a + v_tid) + Int32(768)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(768))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(768))]
  end
  if (((v_l_s_a + v_tid) + Int32(896)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(896))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(896))]
  end
  if (((v_l_s_a + v_tid) + Int32(1024)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1024))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1024))]
  end
  if (((v_l_s_a + v_tid) + Int32(1152)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1152))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1152))]
  end
  if (((v_l_s_a + v_tid) + Int32(1280)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1280))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1280))]
  end
  if (((v_l_s_a + v_tid) + Int32(1408)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1408))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1408))]
  end
  if (((v_l_s_a + v_tid) + Int32(1536)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1536))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1536))]
  end
  if (((v_l_s_a + v_tid) + Int32(1664)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1664))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1664))]
  end
  if (((v_l_s_a + v_tid) + Int32(1792)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1792))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1792))]
  end
  if (((v_l_s_a + v_tid) + Int32(1920)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1920))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1920))]
  end
  v_r_st = (v_r_s_a - v_l_s_a)
  v_r_cnt = (v_r_s_b - v_l_s_b)
  if ((v_l_s_b + v_tid) < v_r_s_b)
   v_smem[(v_r_st + v_tid)] = v_keys[(((v_k + v_loc_b) + v_l_s_b) + v_tid)]
  end
  if (((v_l_s_b + v_tid) + Int32(128)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(128))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(128))]
  end
  if (((v_l_s_b + v_tid) + Int32(256)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(256))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(256))]
  end
  if (((v_l_s_b + v_tid) + Int32(384)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(384))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(384))]
  end
  if (((v_l_s_b + v_tid) + Int32(512)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(512))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(512))]
  end
  if (((v_l_s_b + v_tid) + Int32(640)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(640))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(640))]
  end
  if (((v_l_s_b + v_tid) + Int32(768)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(768))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(768))]
  end
  if (((v_l_s_b + v_tid) + Int32(896)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(896))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(896))]
  end
  if (((v_l_s_b + v_tid) + Int32(1024)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1024))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1024))]
  end
  if (((v_l_s_b + v_tid) + Int32(1152)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1152))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1152))]
  end
  if (((v_l_s_b + v_tid) + Int32(1280)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1280))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1280))]
  end
  if (((v_l_s_b + v_tid) + Int32(1408)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1408))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1408))]
  end
  if (((v_l_s_b + v_tid) + Int32(1536)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1536))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1536))]
  end
  if (((v_l_s_b + v_tid) + Int32(1664)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1664))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1664))]
  end
  if (((v_l_s_b + v_tid) + Int32(1792)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1792))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1792))]
  end
  if (((v_l_s_b + v_tid) + Int32(1920)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1920))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1920))]
  end
  CUDA.sync_threads()
  v_gran = (v_tid << Int32(4))
  v_s_a = sg_find_kth3(ctx ,(v_smem + v_l_st),v_l_cnt,(v_smem + v_r_st),v_r_cnt,v_gran)
  v_s_b = (v_gran - v_s_a)
  v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k4 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k5 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k6 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k7 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k8 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k9 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k10 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k11 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k12 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k13 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k14 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k15 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_warp_id = div(ctx[1],Int32(32))
  v_lane_id = (ctx[1] % Int32(32))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id)] = v_rg_k0
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id)] = v_rg_k2
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id)] = v_rg_k4
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id)] = v_rg_k6
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id)] = v_rg_k8
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id)] = v_rg_k10
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id)] = v_rg_k12
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id)] = v_rg_k14
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id)] = v_rg_k1
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id)] = v_rg_k3
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id)] = v_rg_k5
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id)] = v_rg_k7
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id)] = v_rg_k9
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id)] = v_rg_k11
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id)] = v_rg_k13
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id)] = v_rg_k15
  end
 end
 end
 nothing
end

# bb_comput_l_keys.cuh 9ed7f8a61b4249b6254de2d898bec752b1f0126c27672b795bfe938f9d5abc2f
@inline function sg_kern_copy(ctx,v_srck,v_dstk,v_seg_begins,v_seg_ends,v_bin,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_stride::Int32 = zero(Int32)
 local v_steps::Int32 = zero(Int32)
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if (v_innerbid < v_blk_stat)
  v_tid = ctx[1]
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_stride = sg_upper_power_of_two(ctx ,v_seg_size)
  v_steps = sg_log2(ctx ,div(v_stride,Int32(2048)))
  if ((v_steps & Int32(1)) != 0)
   if (((v_innerbid << Int32(11)) + v_tid) < v_seg_size)
    v_dstk[((v_k + (v_innerbid << Int32(11))) + v_tid)] = v_srck[((v_k + (v_innerbid << Int32(11))) + v_tid)]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(128)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(256)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(384)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(512)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(640)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(768)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(896)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1024)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1152)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1280)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1408)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1536)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1664)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1792)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1920)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))]
   end
  end
 end
 end
 nothing
end

# bb_comput_l.cuh 187e76d3316123387f065733c7b3b45e6708a788cfe62d2862da6ba68aa865c4
@inline function sg_kern_block_sort(ctx,v_key,v_val,v_keyB,v_valB,v_seg_begins,v_seg_ends,v_bin,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 v_tmem=CSpan(CUDA.CuStaticSharedArray(Int32,Int32(2048)),Int32(0))
 local v_bit1::Int32 = zero(Int32)
 local v_bit2::Int32 = zero(Int32)
 local v_bit3::Int32 = zero(Int32)
 local v_bit4::Int32 = zero(Int32)
 local v_bit5::Int32 = zero(Int32)
 local v_warp_lane::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_inner_seg_size::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 local v_grp_start_wp_id::Int32 = zero(Int32)
 local v_grp_start_off::Int32 = zero(Int32)
 local v_lhs_len::Int32 = zero(Int32)
 local v_rhs_len::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_tmp_v0::Int32 = zero(Int32)
 local v_tmp_v1::Int32 = zero(Int32)
 local v_start
 local v_t_v0::UInt64 = zero(UInt64)
 local v_t_v1::UInt64 = zero(UInt64)
 local v_t_v2::UInt64 = zero(UInt64)
 local v_t_v3::UInt64 = zero(UInt64)
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if (v_innerbid < v_blk_stat)
  v_tid = ctx[1]
  v_bit1 = ((v_tid >> Int32(0)) & Int32(0x1))
  v_bit2 = ((v_tid >> Int32(1)) & Int32(0x1))
  v_bit3 = ((v_tid >> Int32(2)) & Int32(0x1))
  v_bit4 = ((v_tid >> Int32(3)) & Int32(0x1))
  v_bit5 = ((v_tid >> Int32(4)) & Int32(0x1))
  v_warp_lane = (ctx[1] & Int32(31))
  v_warp_id = div(ctx[1],Int32(32))
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_k = (v_k + (v_innerbid << Int32(11)))
  v_inner_seg_size = min((v_seg_size - (v_innerbid << Int32(11))),Int32(2048))
  v_rg_k0 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(0)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(0))] : Int32(2147483647))
  v_rg_k1 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(32)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(32))] : Int32(2147483647))
  v_rg_k2 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(64)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(64))] : Int32(2147483647))
  v_rg_k3 = ((((v_warp_lane + (v_warp_id << Int32(7))) + Int32(96)) < v_inner_seg_size) ? v_key[(((v_k + v_warp_lane) + (v_warp_id << Int32(7))) + Int32(96))] : Int32(2147483647))
  if (((v_warp_lane + (v_warp_id << Int32(7))) + Int32(0)) < v_inner_seg_size)
   v_rg_v0 = ((v_warp_lane + (v_warp_id << Int32(7))) + Int32(0))
  end
  if (((v_warp_lane + (v_warp_id << Int32(7))) + Int32(32)) < v_inner_seg_size)
   v_rg_v1 = ((v_warp_lane + (v_warp_id << Int32(7))) + Int32(32))
  end
  if (((v_warp_lane + (v_warp_id << Int32(7))) + Int32(64)) < v_inner_seg_size)
   v_rg_v2 = ((v_warp_lane + (v_warp_id << Int32(7))) + Int32(64))
  end
  if (((v_warp_lane + (v_warp_id << Int32(7))) + Int32(96)) < v_inner_seg_size)
   v_rg_v3 = ((v_warp_lane + (v_warp_id << Int32(7))) + Int32(96))
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k3)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k1 > v_rg_k2)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x3),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x7),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0xf),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_intxn(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1f),v_bit5)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x8),v_bit4)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x4),v_bit3)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x2),v_bit2)
  v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3 = sg_exch_paral(ctx ,v_rg_k0,v_rg_k1,v_rg_k2,v_rg_k3,v_rg_v0,v_rg_v1,v_rg_v2,v_rg_v3,Int32(0x1),v_bit1)
  if (v_rg_k0 > v_rg_k2)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if (v_rg_k1 > v_rg_k3)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if (v_rg_k0 > v_rg_k1)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if (v_rg_k2 > v_rg_k3)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(0))] = v_rg_k0
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(1))] = v_rg_k1
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(2))] = v_rg_k2
  v_smem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(3))] = v_rg_k3
  v_tmem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(0))] = v_rg_v0
  v_tmem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(1))] = v_rg_v1
  v_tmem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(2))] = v_rg_v2
  v_tmem[(((v_warp_id << Int32(7)) + (v_warp_lane << Int32(2))) + Int32(3))] = v_rg_v3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(1)) << Int32(1))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(128)
  v_rhs_len = Int32(128)
  v_gran = (v_warp_lane << Int32(2))
  if ((v_warp_id & Int32(1)) == Int32(0))
   v_gran += Int32(0)
  end
  if ((v_warp_id & Int32(1)) == Int32(1))
   v_gran += Int32(128)
  end
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  if (v_s_a < v_lhs_len)
   v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
  end
  if (v_s_b < (v_lhs_len + v_rhs_len))
   v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
  v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
  v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(2)) << Int32(2))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(256)
  v_rhs_len = Int32(256)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(3)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  if (v_s_a < v_lhs_len)
   v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
  end
  if (v_s_b < (v_lhs_len + v_rhs_len))
   v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
  v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
  v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(3)) << Int32(3))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(512)
  v_rhs_len = Int32(512)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(7)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  if (v_s_a < v_lhs_len)
   v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
  end
  if (v_s_b < (v_lhs_len + v_rhs_len))
   v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  CUDA.sync_threads()
  v_smem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_k0
  v_smem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_k1
  v_smem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_k2
  v_smem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_k3
  v_tmem[((v_grp_start_off + v_gran) + Int32(0))] = v_rg_v0
  v_tmem[((v_grp_start_off + v_gran) + Int32(1))] = v_rg_v1
  v_tmem[((v_grp_start_off + v_gran) + Int32(2))] = v_rg_v2
  v_tmem[((v_grp_start_off + v_gran) + Int32(3))] = v_rg_v3
  CUDA.sync_threads()
  v_grp_start_wp_id = ((v_warp_id >> Int32(4)) << Int32(4))
  v_grp_start_off = (v_grp_start_wp_id * Int32(128))
  v_lhs_len = Int32(1024)
  v_rhs_len = Int32(1024)
  v_gran = (v_warp_lane << Int32(2))
  v_gran += ((v_warp_id & Int32(15)) * Int32(128))
  v_start = (v_smem + v_grp_start_off)
  v_s_a = sg_find_kth3(ctx ,v_start,v_lhs_len,(v_start + v_lhs_len),v_rhs_len,v_gran)
  v_s_b = ((v_lhs_len + v_gran) - v_s_a)
  v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
  if (v_s_a < v_lhs_len)
   v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
  end
  if (v_s_b < (v_lhs_len + v_rhs_len))
   v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_lhs_len) ? v_smem[(v_grp_start_off + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_lhs_len)
    v_tmp_v0 = v_tmem[(v_grp_start_off + v_s_a)]
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < (v_lhs_len + v_rhs_len)) ? v_smem[(v_grp_start_off + v_s_b)] : Int32(2147483647))
   if (v_s_b < (v_lhs_len + v_rhs_len))
    v_tmp_v1 = v_tmem[(v_grp_start_off + v_s_b)]
   end
  end
  v_p = ((v_s_b >= (v_lhs_len + v_rhs_len)) || ((v_s_a < v_lhs_len) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (((v_tid << Int32(2)) + Int32(0)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_rg_k0
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_rg_k1
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_rg_k2
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_inner_seg_size)
   v_keyB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_rg_k3
  end
  if (((v_tid << Int32(2)) + Int32(0)) < v_inner_seg_size)
   v_t_v0 = v_val[(v_k + v_rg_v0)]
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_inner_seg_size)
   v_t_v1 = v_val[(v_k + v_rg_v1)]
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_inner_seg_size)
   v_t_v2 = v_val[(v_k + v_rg_v2)]
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_inner_seg_size)
   v_t_v3 = v_val[(v_k + v_rg_v3)]
  end
  if (((v_tid << Int32(2)) + Int32(0)) < v_inner_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(0))] = v_t_v0
  end
  if (((v_tid << Int32(2)) + Int32(1)) < v_inner_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(1))] = v_t_v1
  end
  if (((v_tid << Int32(2)) + Int32(2)) < v_inner_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(2))] = v_t_v2
  end
  if (((v_tid << Int32(2)) + Int32(3)) < v_inner_seg_size)
   v_valB[((v_k + (v_tid << Int32(2))) + Int32(3))] = v_t_v3
  end
 end
 end
 nothing
end

# bb_comput_l.cuh 187e76d3316123387f065733c7b3b45e6708a788cfe62d2862da6ba68aa865c4
@inline function sg_kern_block_merge(ctx,v_keys,v_vals,v_keysB,v_valsB,v_seg_begins,v_seg_ends,v_bin,v_stride,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 v_smem=CSpan(CUDA.CuStaticSharedArray(Int32,(Int32(128) * Int32(16))),Int32(0))
 local v_k::Int32 = zero(Int32)
 local v_loc_a::Int32 = zero(Int32)
 local v_loc_b::Int32 = zero(Int32)
 local v_cnt_a::Int32 = zero(Int32)
 local v_cnt_b::Int32 = zero(Int32)
 local v_coop::Int32 = zero(Int32)
 local v_coop_bid::Int32 = zero(Int32)
 local v_l_gran::Int32 = zero(Int32)
 local v_r_gran::Int32 = zero(Int32)
 local v_l_s_a::Int32 = zero(Int32)
 local v_l_s_b::Int32 = zero(Int32)
 local v_r_s_a::Int32 = zero(Int32)
 local v_r_s_b::Int32 = zero(Int32)
 local v_l_st::Int32 = zero(Int32)
 local v_l_cnt::Int32 = zero(Int32)
 local v_r_st::Int32 = zero(Int32)
 local v_r_cnt::Int32 = zero(Int32)
 local v_gran::Int32 = zero(Int32)
 local v_s_a::Int32 = zero(Int32)
 local v_s_b::Int32 = zero(Int32)
 local v_p::Bool = zero(Bool)
 local v_rg_k0::Int32 = zero(Int32)
 local v_rg_k1::Int32 = zero(Int32)
 local v_rg_k2::Int32 = zero(Int32)
 local v_rg_k3::Int32 = zero(Int32)
 local v_rg_k4::Int32 = zero(Int32)
 local v_rg_k5::Int32 = zero(Int32)
 local v_rg_k6::Int32 = zero(Int32)
 local v_rg_k7::Int32 = zero(Int32)
 local v_rg_k8::Int32 = zero(Int32)
 local v_rg_k9::Int32 = zero(Int32)
 local v_rg_k10::Int32 = zero(Int32)
 local v_rg_k11::Int32 = zero(Int32)
 local v_rg_k12::Int32 = zero(Int32)
 local v_rg_k13::Int32 = zero(Int32)
 local v_rg_k14::Int32 = zero(Int32)
 local v_rg_k15::Int32 = zero(Int32)
 local v_rg_v0::Int32 = zero(Int32)
 local v_rg_v1::Int32 = zero(Int32)
 local v_rg_v2::Int32 = zero(Int32)
 local v_rg_v3::Int32 = zero(Int32)
 local v_rg_v4::Int32 = zero(Int32)
 local v_rg_v5::Int32 = zero(Int32)
 local v_rg_v6::Int32 = zero(Int32)
 local v_rg_v7::Int32 = zero(Int32)
 local v_rg_v8::Int32 = zero(Int32)
 local v_rg_v9::Int32 = zero(Int32)
 local v_rg_v10::Int32 = zero(Int32)
 local v_rg_v11::Int32 = zero(Int32)
 local v_rg_v12::Int32 = zero(Int32)
 local v_rg_v13::Int32 = zero(Int32)
 local v_rg_v14::Int32 = zero(Int32)
 local v_rg_v15::Int32 = zero(Int32)
 local v_tmp_k0::Int32 = zero(Int32)
 local v_tmp_k1::Int32 = zero(Int32)
 local v_tmp_v0::Int32 = zero(Int32)
 local v_tmp_v1::Int32 = zero(Int32)
 local v_warp_id::Int32 = zero(Int32)
 local v_lane_id::Int32 = zero(Int32)
 local v__t::Int32 = zero(Int32)
 local v__s::Int32 = zero(Int32)
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if ((v_innerbid < v_blk_stat) && (v_stride < v_seg_size))
  v_tid = ctx[1]
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_coop = ((v_stride << Int32(1)) >> Int32(11))
  v_coop_bid = (v_innerbid % v_coop)
  v_loc_a = (div(v_innerbid,v_coop) * (v_stride << Int32(1)))
  v_cnt_a = min(v_stride,(v_seg_size - v_loc_a))
  v_loc_b = min((v_loc_a + v_stride),v_seg_size)
  v_cnt_b = min(v_stride,(v_seg_size - v_loc_b))
  v_l_gran = (v_coop_bid << Int32(11))
  v_r_gran = min(((v_coop_bid + Int32(1)) << Int32(11)),(v_seg_size - v_loc_a))
  v_l_s_a = sg_find_kth3(ctx ,((v_keys + v_k) + v_loc_a),v_cnt_a,((v_keys + v_k) + v_loc_b),v_cnt_b,v_l_gran)
  v_l_s_b = (v_l_gran - v_l_s_a)
  v_r_s_a = sg_find_kth3(ctx ,((v_keys + v_k) + v_loc_a),v_cnt_a,((v_keys + v_k) + v_loc_b),v_cnt_b,v_r_gran)
  v_r_s_b = (v_r_gran - v_r_s_a)
  v_l_st = Int32(0)
  v_l_cnt = (v_r_s_a - v_l_s_a)
  if ((v_l_s_a + v_tid) < v_r_s_a)
   v_smem[(v_l_st + v_tid)] = v_keys[(((v_k + v_loc_a) + v_l_s_a) + v_tid)]
  end
  if (((v_l_s_a + v_tid) + Int32(128)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(128))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(128))]
  end
  if (((v_l_s_a + v_tid) + Int32(256)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(256))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(256))]
  end
  if (((v_l_s_a + v_tid) + Int32(384)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(384))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(384))]
  end
  if (((v_l_s_a + v_tid) + Int32(512)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(512))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(512))]
  end
  if (((v_l_s_a + v_tid) + Int32(640)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(640))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(640))]
  end
  if (((v_l_s_a + v_tid) + Int32(768)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(768))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(768))]
  end
  if (((v_l_s_a + v_tid) + Int32(896)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(896))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(896))]
  end
  if (((v_l_s_a + v_tid) + Int32(1024)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1024))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1024))]
  end
  if (((v_l_s_a + v_tid) + Int32(1152)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1152))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1152))]
  end
  if (((v_l_s_a + v_tid) + Int32(1280)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1280))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1280))]
  end
  if (((v_l_s_a + v_tid) + Int32(1408)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1408))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1408))]
  end
  if (((v_l_s_a + v_tid) + Int32(1536)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1536))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1536))]
  end
  if (((v_l_s_a + v_tid) + Int32(1664)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1664))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1664))]
  end
  if (((v_l_s_a + v_tid) + Int32(1792)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1792))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1792))]
  end
  if (((v_l_s_a + v_tid) + Int32(1920)) < v_r_s_a)
   v_smem[((v_l_st + v_tid) + Int32(1920))] = v_keys[((((v_k + v_loc_a) + v_l_s_a) + v_tid) + Int32(1920))]
  end
  v_r_st = (v_r_s_a - v_l_s_a)
  v_r_cnt = (v_r_s_b - v_l_s_b)
  if ((v_l_s_b + v_tid) < v_r_s_b)
   v_smem[(v_r_st + v_tid)] = v_keys[(((v_k + v_loc_b) + v_l_s_b) + v_tid)]
  end
  if (((v_l_s_b + v_tid) + Int32(128)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(128))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(128))]
  end
  if (((v_l_s_b + v_tid) + Int32(256)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(256))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(256))]
  end
  if (((v_l_s_b + v_tid) + Int32(384)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(384))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(384))]
  end
  if (((v_l_s_b + v_tid) + Int32(512)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(512))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(512))]
  end
  if (((v_l_s_b + v_tid) + Int32(640)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(640))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(640))]
  end
  if (((v_l_s_b + v_tid) + Int32(768)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(768))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(768))]
  end
  if (((v_l_s_b + v_tid) + Int32(896)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(896))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(896))]
  end
  if (((v_l_s_b + v_tid) + Int32(1024)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1024))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1024))]
  end
  if (((v_l_s_b + v_tid) + Int32(1152)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1152))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1152))]
  end
  if (((v_l_s_b + v_tid) + Int32(1280)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1280))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1280))]
  end
  if (((v_l_s_b + v_tid) + Int32(1408)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1408))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1408))]
  end
  if (((v_l_s_b + v_tid) + Int32(1536)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1536))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1536))]
  end
  if (((v_l_s_b + v_tid) + Int32(1664)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1664))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1664))]
  end
  if (((v_l_s_b + v_tid) + Int32(1792)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1792))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1792))]
  end
  if (((v_l_s_b + v_tid) + Int32(1920)) < v_r_s_b)
   v_smem[((v_r_st + v_tid) + Int32(1920))] = v_keys[((((v_k + v_loc_b) + v_l_s_b) + v_tid) + Int32(1920))]
  end
  CUDA.sync_threads()
  v_gran = (v_tid << Int32(4))
  v_s_a = sg_find_kth3(ctx ,(v_smem + v_l_st),v_l_cnt,(v_smem + v_r_st),v_r_cnt,v_gran)
  v_s_b = (v_gran - v_s_a)
  v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
  v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
  if (v_s_a < v_l_cnt)
   v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
  end
  if (v_s_b < v_r_cnt)
   v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k0 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v0 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k1 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v1 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k2 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v2 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k3 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v3 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k4 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v4 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k5 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v5 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k6 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v6 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k7 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v7 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k8 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v8 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k9 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v9 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k10 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v10 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k11 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v11 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k12 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v12 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k13 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v13 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k14 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v14 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  if (v_p != 0)
   v_s_a += Int32(1)
   v_tmp_k0 = ((v_s_a < v_l_cnt) ? v_smem[(v_l_st + v_s_a)] : Int32(2147483647))
   if (v_s_a < v_l_cnt)
    v_tmp_v0 = ((v_loc_a + v_l_s_a) + v_s_a)
   end
  else
   v_s_b += Int32(1)
   v_tmp_k1 = ((v_s_b < v_r_cnt) ? v_smem[(v_r_st + v_s_b)] : Int32(2147483647))
   if (v_s_b < v_r_cnt)
    v_tmp_v1 = ((v_loc_b + v_l_s_b) + v_s_b)
   end
  end
  v_p = ((v_s_b >= v_r_cnt) || ((v_s_a < v_l_cnt) && (v_tmp_k0 <= v_tmp_k1)))
  v_rg_k15 = ((v_p != 0) ? v_tmp_k0 : v_tmp_k1)
  v_rg_v15 = ((v_p != 0) ? v_tmp_v0 : v_tmp_v1)
  v_warp_id = div(ctx[1],Int32(32))
  v_lane_id = (ctx[1] % Int32(32))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x1))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x1))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x1))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x1))
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x1)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x1))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x1))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x1))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x1))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x1))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x1))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x1))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x1))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x1))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x1))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x1))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x1))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x1))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x1))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x1))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x1))
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x2))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x2))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x2))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x2))
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k2
   v_rg_k2 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v2
   v_rg_v2 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v10
   v_rg_v10 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x2)) != 0)
   v__t = v_rg_k13
   v_rg_k13 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v13
   v_rg_v13 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k2,Int32(0x2))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x2))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x2))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x2))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x2))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x2))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x2))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x2))
  v_rg_v2 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v2,Int32(0x2))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x2))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x2))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x2))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x2))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x2))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x2))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x2))
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x4))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x4))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x4))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x4))
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k4
   v_rg_k4 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v4
   v_rg_v4 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k6
   v_rg_k6 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v6
   v_rg_v6 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v12
   v_rg_v12 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k9
   v_rg_k9 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v9
   v_rg_v9 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x4)) != 0)
   v__t = v_rg_k11
   v_rg_k11 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v11
   v_rg_v11 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k4,Int32(0x4))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x4))
  v_rg_k6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k6,Int32(0x4))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x4))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x4))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x4))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x4))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x4))
  v_rg_v4 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v4,Int32(0x4))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x4))
  v_rg_v6 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v6,Int32(0x4))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x4))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x4))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x4))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x4))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x4))
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_v8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v8,Int32(0x8))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x8))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x8))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x8))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x8))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x8))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x8))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x8))
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k8
   v_rg_k8 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v8
   v_rg_v8 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k1
   v_rg_k1 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v1
   v_rg_v1 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k10
   v_rg_k10 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v10
   v_rg_v10 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k3
   v_rg_k3 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v3
   v_rg_v3 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k12
   v_rg_k12 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v12
   v_rg_v12 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k5
   v_rg_k5 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v5
   v_rg_v5 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k14
   v_rg_k14 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v14
   v_rg_v14 = v__s
  end
  if ((v_lane_id & Int32(0x8)) != 0)
   v__t = v_rg_k7
   v_rg_k7 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v7
   v_rg_v7 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k8,Int32(0x8))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x8))
  v_rg_k10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k10,Int32(0x8))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x8))
  v_rg_k12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k12,Int32(0x8))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x8))
  v_rg_k14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k14,Int32(0x8))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x8))
  v_rg_v8 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v8,Int32(0x8))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x8))
  v_rg_v10 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v10,Int32(0x8))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x8))
  v_rg_v12 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v12,Int32(0x8))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x8))
  v_rg_v14 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v14,Int32(0x8))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x8))
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x10))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x10))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x10))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x10))
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k0
   v_rg_k0 = v_rg_k1
   v_rg_k1 = v__t
   v__s = v_rg_v0
   v_rg_v0 = v_rg_v1
   v_rg_v1 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k2
   v_rg_k2 = v_rg_k3
   v_rg_k3 = v__t
   v__s = v_rg_v2
   v_rg_v2 = v_rg_v3
   v_rg_v3 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k4
   v_rg_k4 = v_rg_k5
   v_rg_k5 = v__t
   v__s = v_rg_v4
   v_rg_v4 = v_rg_v5
   v_rg_v5 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k6
   v_rg_k6 = v_rg_k7
   v_rg_k7 = v__t
   v__s = v_rg_v6
   v_rg_v6 = v_rg_v7
   v_rg_v7 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k8
   v_rg_k8 = v_rg_k9
   v_rg_k9 = v__t
   v__s = v_rg_v8
   v_rg_v8 = v_rg_v9
   v_rg_v9 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k10
   v_rg_k10 = v_rg_k11
   v_rg_k11 = v__t
   v__s = v_rg_v10
   v_rg_v10 = v_rg_v11
   v_rg_v11 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k12
   v_rg_k12 = v_rg_k13
   v_rg_k13 = v__t
   v__s = v_rg_v12
   v_rg_v12 = v_rg_v13
   v_rg_v13 = v__s
  end
  if ((v_lane_id & Int32(0x10)) != 0)
   v__t = v_rg_k14
   v_rg_k14 = v_rg_k15
   v_rg_k15 = v__t
   v__s = v_rg_v14
   v_rg_v14 = v_rg_v15
   v_rg_v15 = v__s
  end
  v_rg_k1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k1,Int32(0x10))
  v_rg_k3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k3,Int32(0x10))
  v_rg_k5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k5,Int32(0x10))
  v_rg_k7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k7,Int32(0x10))
  v_rg_k9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k9,Int32(0x10))
  v_rg_k11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k11,Int32(0x10))
  v_rg_k13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k13,Int32(0x10))
  v_rg_k15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_k15,Int32(0x10))
  v_rg_v1 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v1,Int32(0x10))
  v_rg_v3 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v3,Int32(0x10))
  v_rg_v5 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v5,Int32(0x10))
  v_rg_v7 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v7,Int32(0x10))
  v_rg_v9 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v9,Int32(0x10))
  v_rg_v11 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v11,Int32(0x10))
  v_rg_v13 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v13,Int32(0x10))
  v_rg_v15 = CUDA.shfl_xor_sync(UInt32(0xffffffff),v_rg_v15,Int32(0x10))
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id)] = v_rg_k0
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id)] = v_rg_k2
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id)] = v_rg_k4
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id)] = v_rg_k6
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id)] = v_rg_k8
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id)] = v_rg_k10
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id)] = v_rg_k12
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id)] = v_rg_k14
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id)] = v_rg_k1
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id)] = v_rg_k3
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id)] = v_rg_k5
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id)] = v_rg_k7
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id)] = v_rg_k9
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id)] = v_rg_k11
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id)] = v_rg_k13
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id) < v_seg_size)
   v_keysB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id)] = v_rg_k15
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(0)) + v_lane_id)] = v_vals[(v_k + v_rg_v0)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(32)) + v_lane_id)] = v_vals[(v_k + v_rg_v2)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(64)) + v_lane_id)] = v_vals[(v_k + v_rg_v4)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(96)) + v_lane_id)] = v_vals[(v_k + v_rg_v6)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(128)) + v_lane_id)] = v_vals[(v_k + v_rg_v8)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(160)) + v_lane_id)] = v_vals[(v_k + v_rg_v10)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(192)) + v_lane_id)] = v_vals[(v_k + v_rg_v12)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(224)) + v_lane_id)] = v_vals[(v_k + v_rg_v14)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(256)) + v_lane_id)] = v_vals[(v_k + v_rg_v1)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(288)) + v_lane_id)] = v_vals[(v_k + v_rg_v3)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(320)) + v_lane_id)] = v_vals[(v_k + v_rg_v5)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(352)) + v_lane_id)] = v_vals[(v_k + v_rg_v7)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(384)) + v_lane_id)] = v_vals[(v_k + v_rg_v9)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(416)) + v_lane_id)] = v_vals[(v_k + v_rg_v11)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(448)) + v_lane_id)] = v_vals[(v_k + v_rg_v13)]
  end
  if (((((v_innerbid << Int32(11)) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id) < v_seg_size)
   v_valsB[((((v_k + (v_innerbid << Int32(11))) + (v_warp_id << Int32(9))) + Int32(480)) + v_lane_id)] = v_vals[(v_k + v_rg_v15)]
  end
 end
 end
 nothing
end

# bb_comput_l.cuh 187e76d3316123387f065733c7b3b45e6708a788cfe62d2862da6ba68aa865c4
@inline function sg_kern_copy(ctx,v_srck,v_srcv,v_dstk,v_dstv,v_seg_begins,v_seg_ends,v_bin,v_workloads_per_block)
 @inbounds begin
 local v_bin_it::Int32 = zero(Int32)
 local v_innerbid::Int32 = zero(Int32)
 local v_seg_size::Int32 = zero(Int32)
 local v_blk_stat::Int32 = zero(Int32)
 local v_tid::Int32 = zero(Int32)
 local v_k::Int32 = zero(Int32)
 local v_stride::Int32 = zero(Int32)
 local v_steps::Int32 = zero(Int32)
 v_bin_it = ctx[2]
 v_innerbid = ctx[3]
 v_seg_size = (v_seg_ends[v_bin[v_bin_it]] - v_seg_begins[v_bin[v_bin_it]])
 v_blk_stat = div(((v_seg_size + v_workloads_per_block) - Int32(1)),v_workloads_per_block)
 if (v_innerbid < v_blk_stat)
  v_tid = ctx[1]
  v_k = v_seg_begins[v_bin[v_bin_it]]
  v_stride = sg_upper_power_of_two(ctx ,v_seg_size)
  v_steps = sg_log2(ctx ,div(v_stride,Int32(2048)))
  if ((v_steps & Int32(1)) != 0)
   if (((v_innerbid << Int32(11)) + v_tid) < v_seg_size)
    v_dstk[((v_k + (v_innerbid << Int32(11))) + v_tid)] = v_srck[((v_k + (v_innerbid << Int32(11))) + v_tid)]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(128)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(256)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(384)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(512)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(640)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(768)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(896)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1024)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1152)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1280)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1408)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1536)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1664)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1792)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1920)) < v_seg_size)
    v_dstk[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))] = v_srck[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))]
   end
   if (((v_innerbid << Int32(11)) + v_tid) < v_seg_size)
    v_dstv[((v_k + (v_innerbid << Int32(11))) + v_tid)] = v_srcv[((v_k + (v_innerbid << Int32(11))) + v_tid)]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(128)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(128))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(256)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(256))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(384)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(384))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(512)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(512))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(640)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(640))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(768)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(768))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(896)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(896))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1024)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1024))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1152)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1152))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1280)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1280))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1408)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1408))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1536)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1536))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1664)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1664))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1792)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1792))]
   end
   if ((((v_innerbid << Int32(11)) + v_tid) + Int32(1920)) < v_seg_size)
    v_dstv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))] = v_srcv[(((v_k + (v_innerbid << Int32(11))) + v_tid) + Int32(1920))]
   end
  end
 end
 end
 nothing
end