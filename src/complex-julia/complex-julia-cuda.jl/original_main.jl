using CUDA
using Printf

function complex_float_kernel!(check, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    x = Float32(i) * 0.000001f0
    z1 = ComplexF32(sin(x), cos(x))
    z2 = ComplexF32(cos(2.0f0 * x), sin(3.0f0 * x))

    s = UInt8(0)
    s += UInt8(abs(abs(z1 * z2) - abs(z1) * abs(z2)) < 1.0f-3)
    s += UInt8(abs(abs(z1 + z2)^2 - real((z1 + z2) * (conj(z1) + conj(z2)))) < 1.0f-3)
    s += UInt8(abs(abs(z1 - z2)^2 - real((z1 - z2) * (conj(z1) - conj(z2)))) < 1.0f-3)
    s += UInt8(abs(real(z1 * conj(z2) + z2 * conj(z1)) -
                  2.0f0 * (real(z1) * real(z2) + imag(z1) * imag(z2))) < 1.0f-3)
    s += UInt8(abs(abs(conj(z1) / z2) - abs(conj(z1) / conj(z2))) < 1.0f-3)
    @inbounds check[i] = s
    return
end

function complex_double_kernel!(check, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    x = Float64(i) * 0.000001
    z1 = ComplexF64(sin(x), cos(x))
    z2 = ComplexF64(cos(2.0 * x), sin(3.0 * x))

    s = UInt8(0)
    s += UInt8(abs(abs(z1 * z2) - abs(z1) * abs(z2)) < 1.0e-3)
    s += UInt8(abs(abs(z1 + z2)^2 - real((z1 + z2) * (conj(z1) + conj(z2)))) < 1.0e-3)
    s += UInt8(abs(abs(z1 - z2)^2 - real((z1 - z2) * (conj(z1) - conj(z2)))) < 1.0e-3)
    s += UInt8(abs(real(z1 * conj(z2) + z2 * conj(z1)) -
                  2.0 * (real(z1) * real(z2) + imag(z1) * imag(z2))) < 1.0e-3)
    s += UInt8(abs(abs(conj(z1) / z2) - abs(conj(z1) / conj(z2))) < 1.0e-3)
    @inbounds check[i] = s
    return
end

function check_all(check)
    all(v -> v == UInt8(5), Array(check))
end

function timed_run!(kernel, check, n::Int, repeat::Int)
    threads = 256
    blocks = cld(n, threads)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks kernel(check, Int32(n))
    end
    CUDA.synchronize()
    (time_ns() - start) * 1.0e-9 / repeat
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <problem size> <repeat>")
        return 1
    end

    n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    check = CUDA.zeros(UInt8, n)

    @cuda threads=256 blocks=cld(n, 256) complex_float_kernel!(check, Int32(n))
    @cuda threads=256 blocks=cld(n, 256) complex_double_kernel!(check, Int32(n))
    CUDA.synchronize()

    println()
    println("Single-precision complex data type")
    t = timed_run!(complex_float_kernel!, check, n, repeat)
    @printf("Average kernel execution time %f (s)\n", t)
    ok_float = check_all(check)

    t = timed_run!(complex_float_kernel!, check, n, repeat)
    @printf("Average kernel execution time (reference) %f (s)\n", t)
    ok_float &= check_all(check)

    println()
    println("Double-precision complex data type")
    t = timed_run!(complex_double_kernel!, check, n, repeat)
    @printf("Average kernel execution time %f (s)\n", t)
    ok_double = check_all(check)

    t = timed_run!(complex_double_kernel!, check, n, repeat)
    @printf("Average kernel execution time (reference) %f (s)\n", t)
    ok_double &= check_all(check)

    println(ok_float && ok_double ? "PASS" : "FAIL")
    return ok_float && ok_double ? 0 : 1
end

exit(main(ARGS))
