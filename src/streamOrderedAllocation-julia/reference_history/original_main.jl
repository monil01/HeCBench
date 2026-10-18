using CUDA
using Printf
using Random

const REPEAT = 100

function vector_add_kernel!(a, b, c, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds c[idx] = a[idx] + b[idx]
    end
    return
end

function run_add!(a, b, c)
    n = length(a)
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CUDA.zeros(Float32, n)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks vector_add_kernel!(d_a, d_b, d_c, Int32(n))
    CUDA.synchronize()
    copyto!(c, Array(d_c))
    return nothing
end

function check_result(label, a, b, c)
    println("> Checking the results from vectorAddGPU() ...")
    error_norm = 0.0f0
    ref_norm = 0.0f0
    @inbounds for n in eachindex(a)
        ref = a[n] + b[n]
        diff = c[n] - ref
        error_norm += diff * diff
        ref_norm += ref * ref
    end
    ok = sqrt(Float64(error_norm)) / sqrt(Float64(ref_norm)) < 1.0e-6
    ok && println("$label PASSED")
    return ok
end

function allocation_case(label, a, b, c, timing::Bool)
    println("Starting $label()")
    start = time_ns()
    reps = timing ? REPEAT : 1
    for _ in 1:reps
        run_add!(a, b, c)
    end
    elapsed_us = (time_ns() - start) * 1.0e-3
    if timing
        @printf("Total elapsed time: %f (us) over %d iterations\n", elapsed_us, REPEAT)
        return true
    else
        return check_result(label, a, b, c)
    end
end

function main()
    nelem = 33_554_432
    rng = MersenneTwister(1)
    a = rand(rng, Float32, nelem)
    b = rand(rng, Float32, nelem)
    c = Vector{Float32}(undef, nelem)

    ret0 = allocation_case("basicAllocation", a, b, c, false)
    ret1 = allocation_case("basicStreamOrderedAllocation", a, b, c, false)
    ret2 = allocation_case("streamOrderedAllocationPostSync", a, b, c, false)

    allocation_case("basicAllocation", a, b, c, true)
    allocation_case("basicStreamOrderedAllocation", a, b, c, true)
    allocation_case("streamOrderedAllocationPostSync", a, b, c, true)

    return ret0 && ret1 && ret2 ? 0 : 1
end

exit(main())
