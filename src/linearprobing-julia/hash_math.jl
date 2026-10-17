# RetainedhistoricalCUDAjlhashbody, onlyfunction/constantnames changed; sharedonce.
@inline function portable_hash(k::UInt32)
    k ⊻= k >> UInt32(16)
    k *= UInt32(0x85ebca6b)
    k ⊻= k >> UInt32(13)
    k *= UInt32(0xc2b2ae35)
    k ⊻= k >> UInt32(16)
    return k & (ST_CAP - UInt32(1))
end

