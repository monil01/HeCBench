# Standalone CPU-only independent integer audit; never included by GPU entrypoints.
include(joinpath(@__DIR__,"field_math.jl"))
using Random
const AUDIT_P=big(2)^256-big(2)^32-977
function limbs_value(a)
 mod(sum(BigInt(a[i])<<(26*(i-1)) for i in 1:10),AUDIT_P)
end
function from_integer(x)
 fe_storage(ntuple(i->UInt32((x>>(32*(i-1)))&big(0xffffffff)),Val(8)))
end
rng=MersenneTwister(20261018)
values=BigInt[0,1,2,3,AUDIT_P-1,AUDIT_P-2,AUDIT_P÷2,(big(1)<<26)-1,big(1)<<26,(big(1)<<52)-1,(big(1)<<224)-1,(big(1)<<255)-1]
for _ in 1:32
 push!(values,mod(sum(BigInt(rand(rng,UInt64))<<(64*j) for j in 0:3),AUDIT_P))
end
pairs=0;inverse=0
for a in values,b in values
 aa=from_integer(a);bb=from_integer(b)
 @assert limbs_value(fe_mul(aa,bb))==mod(a*b,AUDIT_P)
 global pairs+=1
end
for a in values
 aa=from_integer(a)
 @assert limbs_value(fe_sqr(aa))==mod(a*a,AUDIT_P)
 if a!=0
  @assert limbs_value(fe_inv(aa))==invmod(a,AUDIT_P)
  global inverse+=1
 end
end
println("CPU_LIMB_AUDIT {\"values\":$(length(values)),\"multiplication_pairs\":$pairs,\"squares\":$(length(values)),\"nonzero_inverses\":$inverse,\"correct\":true,\"scope\":\"CPUsameUInt32UInt64limbcodevsindependentBigInt; notGPUprimitiveunitseries\"}")
