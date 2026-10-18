# Native warpBitonicSort stages, shared once across real framework launches.
const COLLISION_STAGES=((1,1,0),(2,2,1),(1,2,0),(4,3,2),(2,3,1),(1,3,0),(8,4,3),(4,4,2),(2,4,1),(1,4,0),(16,4,-1),(8,3,-1),(4,2,-1),(2,1,-1),(1,0,-1))
@inline function collision_sort_value(key,lane)
 for (mask,hi,lo) in COLLISION_STAGES
  dir=((lane>>hi)&Int32(1)) ⊻ (lo<0 ? Int32(0) : ((lane>>lo)&Int32(1)))
  other=CUDA.shfl_xor_sync(typemax(UInt32),key,mask)
  key=(key<other)==(dir!=0) ? other : key
 end
 return key
end
@inline function collision_sort_pair(key,value,lane)
 for (mask,hi,lo) in COLLISION_STAGES
  dir=((lane>>hi)&Int32(1)) ⊻ (lo<0 ? Int32(0) : ((lane>>lo)&Int32(1)))
  other=CUDA.shfl_xor_sync(typemax(UInt32),key,mask);ov=CUDA.shfl_xor_sync(typemax(UInt32),value,mask)
  swap=((key<other)||(key==other&&value<ov))==(dir!=0)
  key=swap ? other : key;value=swap ? ov : value
 end
 return key,value
end
@inline function collision_flag(key,lane)
 sorted=collision_sort_value(key,lane);lower=CUDA.shfl_up_sync(typemax(UInt32),sorted,1)
 return Int32(CUDA.vote_any_sync(typemax(UInt32),lower==sorted&&lane!=0))
end
@inline function collision_mask(key,lane)
 key,origin=collision_sort_pair(key,lane,lane);lower=CUDA.shfl_up_sync(typemax(UInt32),key,1)
 dup=Int32(lower==key&&lane!=0);origin,dup=collision_sort_pair(origin,dup,lane)
 return CUDA.vote_ballot_sync(typemax(UInt32),dup!=0)
end
