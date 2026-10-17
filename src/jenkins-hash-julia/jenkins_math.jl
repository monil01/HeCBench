rot(x::UInt32, k::Int) = (x << k) | (x >> (32 - k))

function mix_vals(a::UInt32, b::UInt32, c::UInt32)
    a -= c; a ⊻= rot(c, 4);  c += b
    b -= a; b ⊻= rot(a, 6);  a += c
    c -= b; c ⊻= rot(b, 8);  b += a
    a -= c; a ⊻= rot(c, 16); c += b
    b -= a; b ⊻= rot(a, 19); a += c
    c -= b; c ⊻= rot(b, 4);  b += a
    return a, b, c
end

function final_vals(a::UInt32, b::UInt32, c::UInt32)
    c ⊻= b; c -= rot(b, 14)
    a ⊻= c; a -= rot(c, 11)
    b ⊻= a; b -= rot(a, 25)
    c ⊻= b; c -= rot(b, 16)
    a ⊻= c; a -= rot(c, 4)
    b ⊻= a; b -= rot(a, 14)
    c ⊻= b; c -= rot(b, 24)
    return a, b, c
end

function mix_remainder(a::UInt32, b::UInt32, c::UInt32,
                       k0::UInt32, k1::UInt32, k2::UInt32, length::UInt32)
    if length == 12
        c += k2; b += k1; a += k0
    elseif length == 11
        c += k2 & 0x00ffffff; b += k1; a += k0
    elseif length == 10
        c += k2 & 0x0000ffff; b += k1; a += k0
    elseif length == 9
        c += k2 & 0x000000ff; b += k1; a += k0
    elseif length == 8
        b += k1; a += k0
    elseif length == 7
        b += k1 & 0x00ffffff; a += k0
    elseif length == 6
        b += k1 & 0x0000ffff; a += k0
    elseif length == 5
        b += k1 & 0x000000ff; a += k0
    elseif length == 4
        a += k0
    elseif length == 3
        a += k0 & 0x00ffffff
    elseif length == 2
        a += k0 & 0x0000ffff
    elseif length == 1
        a += k0 & 0x000000ff
    elseif length == 0
        return c
    end
    _, _, c = final_vals(a, b, c)
    return c
end

function hashlittle_words(keys::AbstractVector{UInt32}, offset::Int, length0::UInt32, initval::UInt32)
    length = length0
    a = UInt32(0xdeadbeef) + length + initval
    b = a
    c = a
    pos = offset
    while length > 12
        a += keys[pos]
        b += keys[pos + 1]
        c += keys[pos + 2]
        a, b, c = mix_vals(a, b, c)
        length -= UInt32(12)
        pos += 3
    end
    return mix_remainder(a, b, c, keys[pos], keys[pos + 1], keys[pos + 2], length)
end

