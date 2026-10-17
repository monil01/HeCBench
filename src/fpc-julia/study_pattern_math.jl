@inline function my_abs_i32(x::Int32)
    mask = x < 0 ? UInt32(0xffffffff) : UInt32(0)
    return (reinterpret(UInt32, x) ⊻ mask) - mask
end

@inline function signed_low32(v::UInt64)
    return reinterpret(Int32, UInt32(v & UInt64(0xffffffff)))
end

@inline function fpc_increment(v::UInt64)
    if v == UInt64(0)
        return Int32(1)
    end

    if my_abs_i32(signed_low32(v)) <= UInt32(0xff)
        return Int32(1)
    end

    if my_abs_i32(signed_low32(v)) <= UInt32(0xffff)
        return Int32(2)
    end

    if (v & UInt64(0xffff)) == UInt64(0)
        return Int32(2)
    end

    lo = Int32(UInt32(v & UInt64(0xffff)))
    hi = Int32(UInt32((v >> 16) & UInt64(0xffff)))
    if my_abs_i32(lo) <= UInt32(0xff) && my_abs_i32(hi) <= UInt32(0xff)
        return Int32(2)
    end

    b0 = v & UInt64(0xff)
    b1 = (v >> 8) & UInt64(0xff)
    b2 = (v >> 16) & UInt64(0xff)
    b3 = (v >> 24) & UInt64(0xff)
    if b0 == b1 && b0 == b2 && b0 == b3
        return Int32(1)
    end

    return Int32(4)
end

@inline function fpc_increment_masks(v::UInt64)
 m1=v==UInt64(0)
 m2=my_abs_i32(signed_low32(v))<=UInt32(0xff)
 m3=my_abs_i32(signed_low32(v))<=UInt32(0xffff)
 m4=(v&UInt64(0xffff))==UInt64(0)
 m5=my_abs_i32(Int32(UInt32(v&UInt64(0xffff))))<=UInt32(0xff)&&my_abs_i32(Int32(UInt32((v>>16)&UInt64(0xffff))))<=UInt32(0xff)
 b0=v&UInt64(0xff);b1=(v>>8)&UInt64(0xff);b2=(v>>16)&UInt64(0xff);b3=(v>>24)&UInt64(0xff)
 m6=b0==b1&&b0==b2&&b0==b3
 return m1 ? Int32(1) : m2 ? Int32(1) : m3 ? Int32(2) : m4 ? Int32(2) : m5 ? Int32(2) : m6 ? Int32(1) : Int32(4)
end
