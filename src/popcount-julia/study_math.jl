const M1 = UInt64(0x5555555555555555)
const M2 = UInt64(0x3333333333333333)
const M4 = UInt64(0x0f0f0f0f0f0f0f0f)
const H01 = UInt64(0x0101010101010101)
const BLOCK_SIZE = 256

@inline function pc1_value(x::UInt64)
    x -= (x >> 1) & M1
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    x += x >> 8
    x += x >> 16
    x += x >> 32
    return Int32(x & UInt64(0x7f))
end

@inline function pc2_value(x::UInt64)
    x -= (x >> 1) & M1
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return Int32((x * H01) >> 56)
end

@inline function pc3_value(x::UInt64)
    count = Int32(0)
    while x != 0
        count += Int32(1)
        x &= x - UInt64(1)
    end
    return count
end

@inline function pc4_value(x::UInt64)
    count = Int32(0)
    i = Int32(0)
    while i < Int32(64)
        count += Int32(x & UInt64(1))
        x >>= 1
        i += Int32(1)
    end
    return count
end

@inline function pc5_value(x::UInt64,lut)
    count=Int32(0)
    for k in 0:7
        count += Int32(@inbounds lut[Int((x >> (8*k)) & UInt64(255))+1])
    end
    return count
end
@inline pc6_value(x::UInt64)=Int32(count_ones(x))
@inline function pc_value(x::UInt64,which::Int32,lut)
    which==1 ? pc1_value(x) : which==2 ? pc2_value(x) : which==3 ? pc3_value(x) : which==4 ? pc4_value(x) : which==5 ? pc5_value(x,lut) : pc6_value(x)
end
