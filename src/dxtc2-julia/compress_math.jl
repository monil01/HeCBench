struct F3
    x::Float32
    y::Float32
    z::Float32
end

@inline f3zero() = F3(0.0f0, 0.0f0, 0.0f0)
@inline Base.:+(a::F3, b::F3) = F3(a.x + b.x, a.y + b.y, a.z + b.z)
@inline Base.:-(a::F3, b::F3) = F3(a.x - b.x, a.y - b.y, a.z - b.z)
@inline Base.:*(a::F3, b::F3) = F3(a.x * b.x, a.y * b.y, a.z * b.z)
@inline Base.:*(a::F3, b::Float32) = F3(a.x * b, a.y * b, a.z * b)
@inline Base.:*(b::Float32, a::F3) = a * b

@inline function first_eigen_vector(m)
    v = F3(1.0f0, 1.0f0, 1.0f0)
    for _ in 1:8
        x = v.x * m[1] + v.y * m[2] + v.z * m[3]
        y = v.x * m[2] + v.y * m[4] + v.z * m[5]
        z = v.x * m[3] + v.y * m[5] + v.z * m[6]
        mx = max(max(x, y), z)
        iv = 1.0f0 / mx
        v = F3(x * iv, y * iv, z * iv)
    end
    return v
end

@inline sat01(x::Float32) = ifelse(isfinite(x), min(max(x, 0.0f0), 1.0f0), 0.0f0)

@inline function round_and_expand(v::F3)
    rx = round(UInt16,sat01(v.x)*31.0f0)
    ry = round(UInt16,sat01(v.y)*63.0f0)
    rz = round(UInt16,sat01(v.z)*31.0f0)
    w = UInt16((rx << 11) | (ry << 5) | rz)
    return F3(Float32(rx) * 0.03227752766457f0,
              Float32(ry) * 0.01583151765563f0,
              Float32(rz) * 0.03227752766457f0), w
end

@inline function alpha4(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? 9.0f0 : b == 0x01 ? 0.0f0 : b == 0x02 ? 6.0f0 : 3.0f0
end

@inline function alpha3(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? 4.0f0 : b == 0x01 ? 0.0f0 : 2.0f0
end

@inline function prod4(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? Int32(0x090000) : b == 0x01 ? Int32(0x000900) :
           b == 0x02 ? Int32(0x040102) : Int32(0x010402)
end

@inline function prod3(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? Int32(0x040000) : b == 0x01 ? Int32(0x000400) :
           b == 0x02 ? Int32(0x040101) : Int32(0x010401)
end

@inline function eval_permutation4(colors, permutation::UInt32, color_sum::F3)
    alphax_sum = f3zero()
    akku = Int32(0)
    for i in 0:15
        bits = permutation >> UInt32(2i)
        alphax_sum = alphax_sum + colors[i+1] * alpha4(bits)
        akku += prod4(bits)
    end
    alpha2_sum = Float32(akku >> 16)
    beta2_sum = Float32((akku >> 8) & 0xff)
    alphabeta_sum = Float32(akku & 0xff)
    betax_sum = color_sum * 9.0f0 - alphax_sum
    factor = 1.0f0 / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum)
    a, start = round_and_expand((alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor)
    b, stop = round_and_expand((betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor)
    e = a * a * alpha2_sum + b * b * beta2_sum +
        (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum) * 2.0f0
    return 0.111111111111f0 * (e.x + e.y + e.z), start, stop
end

@inline function eval_permutation3(colors, permutation::UInt32, color_sum::F3)
    alphax_sum = f3zero()
    akku = Int32(0)
    for i in 0:15
        bits = permutation >> UInt32(2i)
        alphax_sum = alphax_sum + colors[i+1] * alpha3(bits)
        akku += prod3(bits)
    end
    alpha2_sum = Float32(akku >> 16)
    beta2_sum = Float32((akku >> 8) & 0xff)
    alphabeta_sum = Float32(akku & 0xff)
    betax_sum = color_sum * 4.0f0 - alphax_sum
    factor = 1.0f0 / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum)
    a, start = round_and_expand((alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor)
    b, stop = round_and_expand((betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor)
    e = a * a * alpha2_sum + b * b * beta2_sum +
        (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum) * 2.0f0
    return 0.25f0 * (e.x + e.y + e.z), start, stop
end


@inline loadcolor(w,i)=F3(w[3i+1],w[3i+2],w[3i+3])
@inline function storecolor!(w,i,c)
 @inbounds w[3i+1]=c.x;w[3i+2]=c.y;w[3i+3]=c.z
 nothing
end
