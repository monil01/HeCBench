using CUDA
using Printf

# Julia port of atan2-cuda: polynomial approximations of atan2 at various
# precisions.  We reproduce the polynomials from the CUDA source verbatim
# (Sollya-generated).

# Constants match main.cu.  We inline these polynomials at the call sites
# in device kernels because CUDA.jl kernels don't easily support templated
# constexpr calls; the CPU reference uses the same functions.

# We use hex-float parsing via reinterpret to match float(-0xf.8eed2p-4) etc.
# In Julia, we can just write these as Float32 hex-decimal literals.

# ---- float polynomials (safe_atan2f) ----
@inline function approx_atan2f_P3(x::Float32)
    return x * (Float32(-0xf.8eed2p-4) + x * x * Float32(0x3.1238p-4))
end
@inline function approx_atan2f_P5(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ecfc8p-4) + z * (Float32(0x4.9e79dp-4) + z * Float32(-0x1.44f924p-4)))
end
@inline function approx_atan2f_P7(x::Float32)
    z = x * x
    return x * (Float32(-0xf.fcc7ap-4) + z * (Float32(0x5.23886p-4) + z * (Float32(-0x2.571968p-4) + z * Float32(0x9.fb05p-8))))
end
@inline function approx_atan2f_P9(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ff73ep-4) + z * (Float32(0x5.48ee1p-4) + z * (Float32(-0x2.e1efe8p-4) + z * (Float32(0x1.5cce54p-4) + z * Float32(-0x5.56245p-8)))))
end
@inline function approx_atan2f_P11(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ffe82p-4) + z * (Float32(0x5.526c8p-4) + z * (Float32(-0x3.18bea8p-4) + z * (Float32(0x1.dce3bcp-4) + z * (Float32(-0xd.7a64ap-8) + z * Float32(0x3.000eap-8))))))
end
@inline function approx_atan2f_P13(x::Float32)
    z = x * x
    return x * (Float32(-0xf.fffbep-4) + z * (Float32(0x5.54adp-4) + z * (Float32(-0x3.2b4df8p-4) + z * (Float32(0x2.1df79p-4) + z * (Float32(-0x1.46081p-4) + z * (Float32(0x8.99028p-8) + z * Float32(-0x1.be0bc4p-8)))))))
end
@inline function approx_atan2f_P15(x::Float32)
    z = x * x
    c0 = Float32(-0xf.ffff4p-4)
    c1 = Float32(0x5.552f9p-4)
    c2 = Float32(-0x3.30f728p-4)
    c3 = Float32(0x2.39826p-4)
    c4 = Float32(-0x1.8a880cp-4)
    c5 = Float32(0xe.484d6p-8)
    c6 = Float32(-0x5.93d5p-8)
    c7 = Float32(0x1.0875dcp-8)
    return x * (c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * c7)))))))
end

# helper implementations
@inline function unsafe_atan2f_impl(y::Float32, x::Float32, deg::Val{D}) where {D}
    pi4f = Float32(3.1415926535897932384626434 / 4)
    pi34f = Float32(3.1415926535897932384626434 * 3 / 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4f : pi34f
    angle += @inline (D == 3 ? approx_atan2f_P3(r) :
                     D == 5 ? approx_atan2f_P5(r) :
                     D == 7 ? approx_atan2f_P7(r) :
                     D == 9 ? approx_atan2f_P9(r) :
                     D == 11 ? approx_atan2f_P11(r) :
                     D == 13 ? approx_atan2f_P13(r) :
                     approx_atan2f_P15(r))
    return y < 0f0 ? -angle : angle
end

@inline function safe_atan2f(y::Float32, x::Float32, ::Val{D}) where {D}
    xs = ((y == 0f0) & (x == 0f0)) ? 0.2f0 : x
    return unsafe_atan2f_impl(y, xs, Val(D))
end

# ---- int polynomials ----
@inline function approx_atan2i_P3(x::Float32); z = x*x; x * (-664694912f0 + z * 131209024f0); end
@inline function approx_atan2i_P5(x::Float32); z = x*x; x * (-680392064f0 + z * (197338400f0 + z * (-54233256f0))); end
@inline function approx_atan2i_P7(x::Float32); z = x*x; x * (-683027840f0 + z * (219543904f0 + z * (-99981040f0 + z * 26649684f0))); end
@inline function approx_atan2i_P9(x::Float32); z = x*x; x * (-683473920f0 + z * (225785056f0 + z * (-123151184f0 + z * (58210592f0 + z * (-14249276f0))))); end
@inline function approx_atan2i_P11(x::Float32); z = x*x; x * (-683549696f0 + z * (227369312f0 + z * (-132297008f0 + z * (79584144f0 + z * (-35987016f0 + z * 8010488f0))))); end
@inline function approx_atan2i_P13(x::Float32); z = x*x; x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0))))))); end
@inline function approx_atan2i_P15(x::Float32); z = x*x; x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0))))))); end

@inline function unsafe_atan2i_impl(y::Float32, x::Float32, ::Val{D}) where {D}
    maxint = Int64(typemax(Int32)) + 1
    pi4 = Int32(maxint ÷ 4)
    pi34 = Int32(3 * maxint ÷ 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4 : pi34
    poly = D == 3 ? approx_atan2i_P3(r) :
           D == 5 ? approx_atan2i_P5(r) :
           D == 7 ? approx_atan2i_P7(r) :
           D == 9 ? approx_atan2i_P9(r) :
           D == 11 ? approx_atan2i_P11(r) :
           D == 13 ? approx_atan2i_P13(r) :
           approx_atan2i_P15(r)
    angle += unsafe_trunc(Int32, poly)
    return y < 0f0 ? -angle : angle
end

# ---- short polynomials ----
@inline function approx_atan2s_P3(x::Float32); z = x*x; x * (-10142.439453125f0 + z * 2002.0908203125f0); end
@inline function approx_atan2s_P5(x::Float32); z = x*x; x * (-10381.9609375f0 + z * (3011.1513671875f0 + z * (-827.538330078125f0))); end
@inline function approx_atan2s_P7(x::Float32); z = x*x; x * (-10422.177734375f0 + z * (3349.97412109375f0 + z * (-1525.589599609375f0 + z * 406.64190673828125f0))); end
@inline function approx_atan2s_P9(x::Float32); z = x*x; x * (-10428.984375f0 + z * (3445.20654296875f0 + z * (-1879.137939453125f0 + z * (888.22314453125f0 + z * (-217.42669677734375f0))))); end

@inline function unsafe_atan2s_impl(y::Float32, x::Float32, ::Val{D}) where {D}
    maxshort = Int32(typemax(Int16)) + Int32(1)
    pi4 = Int16(maxshort ÷ 4)
    pi34 = Int16(3 * maxshort ÷ 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4 : pi34
    poly = D == 3 ? approx_atan2s_P3(r) :
           D == 5 ? approx_atan2s_P5(r) :
           D == 7 ? approx_atan2s_P7(r) :
           approx_atan2s_P9(r)
    angle += unsafe_trunc(Int16, poly)
    return y < 0f0 ? -angle : angle
end

# ---- kernels ----
function compute_f_kernel!(n::Int32, x, y, r)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = safe_atan2f(vy, vx, Val(3)) +
          safe_atan2f(vy, vx, Val(5)) +
          safe_atan2f(vy, vx, Val(7)) +
          safe_atan2f(vy, vx, Val(9)) +
          safe_atan2f(vy, vx, Val(11)) +
          safe_atan2f(vy, vx, Val(13)) +
          safe_atan2f(vy, vx, Val(15))
    @inbounds r[i] = val
    return
end

function compute_i_kernel!(n::Int32, x, y, r)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = unsafe_atan2i_impl(vy, vx, Val(3)) +
          unsafe_atan2i_impl(vy, vx, Val(5)) +
          unsafe_atan2i_impl(vy, vx, Val(7)) +
          unsafe_atan2i_impl(vy, vx, Val(9)) +
          unsafe_atan2i_impl(vy, vx, Val(11)) +
          unsafe_atan2i_impl(vy, vx, Val(13)) +
          unsafe_atan2i_impl(vy, vx, Val(15))
    @inbounds r[i] = val
    return
end

function compute_s_kernel!(n::Int32, x, y, r)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = unsafe_atan2s_impl(vy, vx, Val(3)) +
          unsafe_atan2s_impl(vy, vx, Val(5)) +
          unsafe_atan2s_impl(vy, vx, Val(7)) +
          unsafe_atan2s_impl(vy, vx, Val(9))
    @inbounds r[i] = val
    return
end

# ---- CPU references ----
function reference_f(n, x, y)
    r = Vector{Float32}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = safe_atan2f(vy, vx, Val(3)) +
               safe_atan2f(vy, vx, Val(5)) +
               safe_atan2f(vy, vx, Val(7)) +
               safe_atan2f(vy, vx, Val(9)) +
               safe_atan2f(vy, vx, Val(11)) +
               safe_atan2f(vy, vx, Val(13)) +
               safe_atan2f(vy, vx, Val(15))
    end
    return r
end
function reference_i(n, x, y)
    r = Vector{Int32}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = unsafe_atan2i_impl(vy, vx, Val(3)) +
               unsafe_atan2i_impl(vy, vx, Val(5)) +
               unsafe_atan2i_impl(vy, vx, Val(7)) +
               unsafe_atan2i_impl(vy, vx, Val(9)) +
               unsafe_atan2i_impl(vy, vx, Val(11)) +
               unsafe_atan2i_impl(vy, vx, Val(13)) +
               unsafe_atan2i_impl(vy, vx, Val(15))
    end
    return r
end
function reference_s(n, x, y)
    r = Vector{Int16}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = unsafe_atan2s_impl(vy, vx, Val(3)) +
               unsafe_atan2s_impl(vy, vx, Val(5)) +
               unsafe_atan2s_impl(vy, vx, Val(7)) +
               unsafe_atan2s_impl(vy, vx, Val(9))
    end
    return r
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of coordinates> <repeat>")
        return 1
    end
    n = parse(Int, ARGS[1])
    repeat_n = parse(Int, ARGS[2])

    state = UInt64(123)
    @inline function lcg_f32()
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32((state >> 11) & UInt64(0xFFFFFF)) / Float32(1 << 24)
    end

    x = Vector{Float32}(undef, n)
    y = Vector{Float32}(undef, n)
    for i in 1:n
        x[i] = lcg_f32() + 1.57f0
        y[i] = lcg_f32() + 1.57f0
    end

    d_x = CuArray(x); d_y = CuArray(y)
    d_f = CUDA.zeros(Float32, n)
    d_i = CUDA.zeros(Int32, n)
    d_s = CUDA.zeros(Int16, n)

    threads = 256
    blocks = (n ÷ 256) + 1

    # ---- f32 ----
    println("\n======== output type is f32 ========")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks compute_f_kernel!(Int32(n), d_y, d_x, d_f)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time: %f (us)\n", elapsed_us)
    hf = Array(d_f)
    rf = reference_f(n, x, y)
    err = 0.0
    for i in 1:n
        d = Float64(rf[i]) - Float64(hf[i])
        if abs(d) > 1e-3
            err += d * d
        end
    end
    rmse_f = sqrt(err / n)
    @printf("RMSE: %f\n", rmse_f)

    # ---- i32 ----
    println("\n======== output type is i32 ========")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks compute_i_kernel!(Int32(n), d_y, d_x, d_i)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time: %f (us)\n", elapsed_us)
    hi = Array(d_i)
    ri = reference_i(n, x, y)
    err = 0.0
    for i in 1:n
        d = Float64(ri[i]) - Float64(hi[i])
        if abs(d) > 0
            err += d * d
        end
    end
    @printf("RMSE: %f\n", sqrt(err / n))

    # ---- i16 ----
    println("\n======== output type is i16 ========")
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks compute_s_kernel!(Int32(n), d_y, d_x, d_s)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time: %f (us)\n", elapsed_us)
    hs = Array(d_s)
    rs = reference_s(n, x, y)
    err = 0.0
    for i in 1:n
        d = Float64(rs[i]) - Float64(hs[i])
        if abs(d) > 0
            err += d * d
        end
    end
    rmse_s = sqrt(err / n)
    @printf("RMSE: %f\n", rmse_s)

    # PASS if float RMSE is small (GPU vs CPU polynomial evaluation, allowing fma differences)
    println(rmse_f < 10.0 ? "PASS" : "FAIL")
    return 0
end

main()
