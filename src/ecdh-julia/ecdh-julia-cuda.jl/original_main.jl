using CUDA
using Printf

const P_X = Int32(5)
const P_Y = Int32(1)
const MODULUS = UInt32(17)
const A_COEFF = Int32(2)

@inline function ext_euclidian_alg(a::Int32, b::UInt32)
    x = Int32(1)
    y = Int32(0)
    x1 = Int32(0)
    y1 = Int32(1)
    a1 = a
    b1 = Int32(b)

    while b1 != Int32(0)
        q = a1 ÷ b1
        s = x1
        t = x - q * x1
        x = s
        x1 = t
        s = y1
        t = y - q * y1
        y = s
        y1 = t
        s = b1
        t = a1 - q * b1
        a1 = s
        b1 = t
    end

    return a1, x, y
end

@inline function make_positive(a::Int32, m::UInt32)
    v = a
    mm = Int32(m)
    while v < Int32(0)
        v += mm
    end
    return Int32(v % mm)
end

@inline function find_inverse(a::Int32, m::UInt32)
    _, t, _ = ext_euclidian_alg(a, m)
    return make_positive(t, m)
end

@inline function point_addition(m::UInt32, x1::Int32, y1::Int32, x2::Int32, y2::Int32)
    temp = make_positive(x2 - x1, m)
    slope = make_positive((y2 - y1) * find_inverse(temp, m), m)
    x3 = make_positive(slope * slope - x1 - x2, m)
    y3 = make_positive(slope * (x1 - x3) - y1, m)
    return x3, y3
end

@inline function point_doubling(m::UInt32, a::Int32, x1::Int32, y1::Int32)
    slope = (Int32(3) * x1 * x1 + a) * find_inverse(Int32(2) * y1, m)
    x3 = make_positive(slope * slope - Int32(2) * x1, m)
    y3 = make_positive(slope * (x1 - x3) - y1, m)
    return x3, y3
end

@inline function first_set_bit(n::Int32)
    for i in Int32(31):-Int32(1):Int32(0)
        if ((Int32(1) << i) & n) != Int32(0)
            return i
        end
    end
    return Int32(0)
end

@inline function make_pk_fast(sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32)
    tx = px
    ty = py

    for i in (first_set_bit(sk) - Int32(1)):-Int32(1):Int32(0)
        tx, ty = point_doubling(m, a, tx, ty)
        if ((Int32(1) << i) & sk) != Int32(0)
            tx, ty = point_addition(m, tx, ty, px, py)
        end
    end

    return tx, ty
end

@inline function make_pk_slow(sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32)
    tx, ty = point_doubling(m, a, px, py)
    remaining = sk - Int32(2)

    while remaining > Int32(0)
        tx, ty = point_addition(m, tx, ty, px, py)
        remaining -= Int32(1)
    end

    return tx, ty
end

function slow_kernel!(pk_x, pk_y, sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32, num_pk::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= num_pk
        x, y = make_pk_slow(sk, px, py, m, a)
        @inbounds pk_x[i] = x
        @inbounds pk_y[i] = y
    end
    return
end

function fast_kernel!(pk_x, pk_y, sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32, num_pk::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= num_pk
        x, y = make_pk_fast(sk, px, py, m, a)
        @inbounds pk_x[i] = x
        @inbounds pk_y[i] = y
    end
    return
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <positive number of keys> <repeat>")
        return 1
    end

    num_pk = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])
    blocks = cld(num_pk, 256)

    d_pk_x = CUDA.zeros(Int32, num_pk)
    d_pk_y = CUDA.zeros(Int32, num_pk)

    start_ns = time_ns()
    for _ in 1:repeat
        @cuda threads=256 blocks=blocks slow_kernel!(d_pk_x, d_pk_y, Int32(18), P_X, P_Y, MODULUS, A_COEFF, Int32(num_pk))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start_ns) * 1.0e-9 / repeat
    @printf("Average time (slow kernel): %f s\n", elapsed_s)

    pk_slow_x = Array(d_pk_x)
    pk_slow_y = Array(d_pk_y)

    start_ns = time_ns()
    for _ in 1:repeat
        @cuda threads=256 blocks=blocks fast_kernel!(d_pk_x, d_pk_y, Int32(18), P_X, P_Y, MODULUS, A_COEFF, Int32(num_pk))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start_ns) * 1.0e-9 / repeat
    @printf("Average time (fast kernel): %f s\n", elapsed_s)

    pk_fast_x = Array(d_pk_x)
    pk_fast_y = Array(d_pk_y)

    println((pk_slow_x == pk_fast_x && pk_slow_y == pk_fast_y) ? "PASS" : "FAIL")
    return 0
end

exit(main())
