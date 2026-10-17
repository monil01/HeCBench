# Source-faithful table generators and Moro helper from archived genuine CUDA.jl.
const QRNG_DIMENSIONS = 3
const QRNG_RESOLUTION = 31
const N = UInt32(1048576)
const INT_SCALE_F32 = Float32(1.0) / Float32(0x80000001)
const UINT32_MAX_F32 = Float32(0xffffffff)

@inline function moro_inv_cnd_gpu(x_in::UInt32)
    a1 = Float32(2.50662823884)
    a2 = Float32(-18.61500062529)
    a3 = Float32(41.39119773534)
    a4 = Float32(-25.44106049637)
    b1 = Float32(-8.4735109309)
    b2 = Float32(23.08336743743)
    b3 = Float32(-21.06224101826)
    b4 = Float32(3.13082909833)
    c1 = Float32(0.337475482272615)
    c2 = Float32(0.976169019091719)
    c3 = Float32(0.160797971491821)
    c4 = Float32(2.76438810333863e-2)
    c5 = Float32(3.8405729373609e-3)
    c6 = Float32(3.951896511919e-4)
    c7 = Float32(3.21767881768e-5)
    c8 = Float32(2.888167364e-7)
    c9 = Float32(3.960315187e-7)

    x = x_in
    negate = false
    if x >= UInt32(0x80000000)
        x = UInt32(0xffffffff) - x
        negate = true
    end

    x1 = Float32(1.0) / UINT32_MAX_F32
    x2 = x1 / Float32(2.0)
    p1 = Float32(x) * x1 + x2
    p2 = p1 - Float32(0.5)
    if p2 > Float32(-0.42)
        z = p2 * p2
        z = p2 * (((a4 * z + a3) * z + a2) * z + a1) /
            ((((b4 * z + b3) * z + b2) * z + b1) * z + Float32(1.0))
    else
        z = log(-log(p1))
        z = -(c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))))
    end
    return negate ? -z : z
end

function generate_polynomials(primitive::Bool)
    buffer = Vector{Int32}(undef, QRNG_DIMENSIONS)
    buffer[1] = Int32(0x2)
    l = Int32(0)
    for n in 2:QRNG_DIMENSIONS
        p1 = buffer[n - 1] + Int32(1)
        while true
            e_p1 = Int32(30)
            while (p1 & (Int32(1) << e_p1)) == 0
                e_p1 -= Int32(1)
            end
            p2 = Int32(0)
            for i in 1:(n - 1)
                e_b = e_p1
                while (buffer[i] & (Int32(1) << e_b)) == 0
                    e_b -= Int32(1)
                end
                e_p2 = e_p1
                p2 = (buffer[i] << (e_p2 - e_b)) ⊻ p1
                while p2 >= buffer[i]
                    while (p2 & (Int32(1) << e_p2)) == 0
                        e_p2 -= Int32(1)
                    end
                    p2 = (buffer[i] << (e_p2 - e_b)) ⊻ p2
                end
                p2 == 0 && break
            end
            if p2 != 0
                e_p2 = Int32(0)
                if primitive
                    j = ~(-Int32(1) << (e_p1 + Int32(1)))
                    e_b = (Int32(1) << e_p1) | Int32(0x1)
                    p2 = e_b
                    e_p2 = (Int32(1) << e_p1) - Int32(2)
                    while e_p2 > 0
                        p2 <<= Int32(1)
                        i = p2 & p1
                        i = (i & Int32(0x55555555)) + ((i >> Int32(1)) & Int32(0x55555555))
                        i = (i & Int32(0x33333333)) + ((i >> Int32(2)) & Int32(0x33333333))
                        i = (i & Int32(0x07070707)) + ((i >> Int32(4)) & Int32(0x07070707))
                        p2 |= (i % Int32(255)) & Int32(1)
                        ((p2 & j) == e_b) && break
                        e_p2 -= Int32(1)
                    end
                end
                if e_p2 == 0
                    buffer[n] = p1
                    l += e_p1
                    break
                end
            end
            p1 += Int32(1)
        end
    end
    return buffer, l + Int32(1)
end

function generate_cj()
    cjn = zeros(Int64, 63, QRNG_DIMENSIONS)
    buffer, l_total = generate_polynomials(false)
    polynomials = Vector{Int32}(undef, Int(l_total + Int32(2 * QRNG_DIMENSIONS + 1)))
    l = 1
    for n in 1:QRNG_DIMENSIONS
        p1 = buffer[n]
        e_p1 = Int32(30)
        while (p1 & (Int32(1) << e_p1)) == 0
            e_p1 -= Int32(1)
        end
        polynomials[l] = Int32(1); l += 1
        e_p1 -= Int32(1)
        while e_p1 >= 0
            polynomials[l] = (p1 >> e_p1) & Int32(1)
            l += 1
            e_p1 -= Int32(1)
        end
        polynomials[l] = Int32(-1); l += 1
    end
    polynomials[l] = Int32(-1)

    p_index = 1
    d = 1
    while polynomials[p_index] != -1
        e = 0
        while polynomials[p_index + e + 1] != -1
            e += 1
        end
        b = zeros(Int32, 1024)
        v = zeros(Int32, 1024)
        b[1024] = Int32(1)
        m = 0
        u = e
        for j in 62:-1:0
            if u == e
                u = 0
                t = copy(b)
                m1 = m
                m += e
                fill!(b, 0)
                for i in 0:m
                    acc = Int32(0)
                    ip_start = e - (m - i)
                    it = m1
                    ip = ip_start
                    while ip <= e && it >= 0
                        if ip >= 0
                            acc ⊻= polynomials[p_index + ip] & t[1024 - m1 + it]
                        end
                        ip += 1
                        it -= 1
                    end
                    b[1024 - m + i] = acc
                end

                for i in 0:(m1 - 1)
                    v[1024 - (63 + e - 2) + i] = Int32(0)
                end
                for i in m1:(m - 1)
                    v[1024 - (63 + e - 2) + i] = Int32(1)
                end
                for i in m:(63 + e - 2)
                    vv = Int32(0)
                    for it in 1:m
                        vv ⊻= v[1024 - (63 + e - 2) + i - it] & b[1024 - m + it]
                    end
                    v[1024 - (63 + e - 2) + i] = vv
                end
            end
            for i in 0:62
                cjn[i + 1, d] |= Int64(v[1024 - (63 + e - 2) + i + u]) << j
            end
            u += 1
        end
        d += 1
        p_index += e + 2
    end
    return cjn
end

function init_quasirandom_generator()
    cjn = generate_cj()
    table = Vector{UInt32}(undef, QRNG_DIMENSIONS * QRNG_RESOLUTION)
    for dim in 0:(QRNG_DIMENSIONS - 1)
        for bit in 0:(QRNG_RESOLUTION - 1)
            table[dim * QRNG_RESOLUTION + bit + 1] =
                UInt32((cjn[bit + 1, dim + 1] >> 32) & Int64(0x7fffffff))
        end
    end
    return table, cjn
end

function get_quasirandom_value63(cjn, i::Int64, dim::Int)
    int63_scale = 1.0 / Float64(0x8000000000000001)
    result = Int64(0)
    for bit in 0:62
        if (i & Int64(1)) != 0
            result ⊻= cjn[bit + 1, dim + 1]
        end
        i >>= 1
    end
    return Float64(result + 1) * int63_scale
end

function moro_inv_cnd_cpu(x_in::UInt32)
    a1 = 2.50662823884
    a2 = -18.61500062529
    a3 = 41.39119773534
    a4 = -25.44106049637
    b1 = -8.4735109309
    b2 = 23.08336743743
    b3 = -21.06224101826
    b4 = 3.13082909833
    c1 = 0.337475482272615
    c2 = 0.976169019091719
    c3 = 0.160797971491821
    c4 = 2.76438810333863e-2
    c5 = 3.8405729373609e-3
    c6 = 3.951896511919e-4
    c7 = 3.21767881768e-5
    c8 = 2.888167364e-7
    c9 = 3.960315187e-7

    x = x_in
    negate = false
    if x >= UInt32(0x80000000)
        x = UInt32(0xffffffff) - x
        negate = true
    end
    x1 = 1.0 / Float64(0xffffffff)
    x2 = x1 / 2.0
    p1 = Float64(x) * x1 + x2
    p2 = p1 - 0.5
    if p2 > -0.42
        z = p2 * p2
        z = p2 * (((a4 * z + a3) * z + a2) * z + a1) /
            ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0)
    else
        z = log(-log(p1))
        z = -(c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))))
    end
    return negate ? -z : z
end

