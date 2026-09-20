using CUDA
using Printf

const CURAND = CUDA.CURAND
const LIBCURAND = getfield(CURAND, :libcurand)

function create_generator(host::Bool)
    gen_ref = Ref{CURAND.curandGenerator_t}()
    rng = CURAND.CURAND_RNG_PSEUDO_MRG32K3A
    if host
        CURAND.curandCreateGeneratorHost(gen_ref, rng)
    else
        CURAND.curandCreateGenerator(gen_ref, rng)
    end
    return gen_ref[]
end

function configure_generator!(gen)
    CURAND.curandSetGeneratorOffset(gen, UInt64(0))
    CURAND.curandSetGeneratorOrdering(gen, CURAND.CURAND_ORDERING_PSEUDO_BEST)
    CURAND.curandSetPseudoRandomGeneratorSeed(gen, UInt64(1234))
    return gen
end

function run_on_device(n::Int)
    d_data = CuArray{Float32}(undef, n)
    gen = configure_generator!(create_generator(false))
    CURAND.curandGenerateUniform(gen, d_data, Csize_t(n))
    h_data = Vector{Float32}(undef, n)
    copyto!(h_data, d_data)
    CURAND.curandDestroyGenerator(gen)
    return h_data
end

function run_on_host(n::Int)
    h_data = Vector{Float32}(undef, n)
    gen = configure_generator!(create_generator(true))
    status = ccall((:curandGenerateUniform, LIBCURAND), Cint,
                   (CURAND.curandGenerator_t, Ptr{Float32}, Csize_t),
                   gen, pointer(h_data), Csize_t(n))
    status == Int(CURAND.CURAND_STATUS_SUCCESS) || error("curandGenerateUniform host failed: $status")
    CURAND.curandDestroyGenerator(gen)
    return h_data
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <number of pseudorandom numbers to generate> <repeat>")
        exit(1)
    end
    n = parse(Int, ARGS[1])
    repeat = parse(Int, ARGS[2])

    h_data = run_on_host(n)
    d_data = run_on_device(n)

    t0 = time_ns()
    for _ in 1:repeat
        h_data = run_on_host(n)
    end
    host_ns = time_ns() - t0
    @printf("Average execution time on host: %f (us)\n", (host_ns * 1e-3) / repeat)

    CUDA.synchronize()
    t1 = time_ns()
    for _ in 1:repeat
        d_data = run_on_device(n)
    end
    CUDA.synchronize()
    dev_ns = time_ns() - t1
    @printf("Average execution time on device: %f (us)\n", (dev_ns * 1e-3) / repeat)

    ok = true
    for i in 1:n
        if abs(h_data[i] - d_data[i]) > 1f-3
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    ok || exit(1)
end

main()
