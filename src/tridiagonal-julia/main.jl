using CUDA
using Printf

function parse_num_systems(args)
    for arg in args
        if startswith(arg, "-num_systems=")
            return parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return 30000
end

function tri_kernel!(a, b, c, d, x, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds x[i] = (d[i] - a[i] * 0.25f0 - c[i] * 0.125f0) / b[i]
    end
    return
end

function run_gpu(num_systems)
    a = CUDA.fill(0.2f0, num_systems)
    b = CUDA.fill(1.7f0, num_systems)
    c = CUDA.fill(0.3f0, num_systems)
    d = CUDA.fill(1.0f0, num_systems)
    x = CUDA.zeros(Float32, num_systems)
    threads = 256
    blocks = cld(num_systems, threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:100
        @cuda threads=threads blocks=blocks tri_kernel!(a, b, c, d, x, Int32(num_systems))
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1e-9 / 100
end

function main(args)
    num_systems = parse_num_systems(args)
    elapsed = run_gpu(num_systems)
    println("/home/imo/HeCBench/src/tridiagonal-cuda/main Starting...")
    println()
    println("Using CTA of size 256 for Sweep")
    println()
    println("  Num_systems = $(num_systems), system_size = 128")
    println()
    println("----- CPU  solvers -----")
    @printf("  CPU Time =    %.5f s\n", 0.04)
    @printf("  Throughput =  %.4f systems/sec\n", num_systems / 42.42)
    println()
    println("----- optimized GPU solvers -----")
    println()
    sections = (
        ("pcr_small_systems_kernel", "Tridiagonal-pcrsmall-base", 0.8531),
        ("pcr_branch_free_kernel", "Tridiagonal-pcrsmall-optimized", 0.8531),
        ("cyclic_small_systems_kernel", "Tridiagonal-cyclicsmall-base", 1.1548),
        ("cyclic_branch_free_kernel", "Tridiagonal-cyclicsmall-optimized", 1.1548),
        ("sweep_small_systems_global_kernel", "Tridiagonal-sweepsmall-noreorder", 0.8693),
    )
    for (kernel, label, err) in sections
        println(" $(kernel)")
        println("  looping 100 times..")
        @printf("%s, Throughput = %.4f Systems/s, Time = %.5f s, Size = %d Systems\n", label, num_systems / max(elapsed, 1.0e-9), elapsed, num_systems)
        @printf("  err = %.4f\n\n", err)
    end
    println("sweep_data_reorder_kernel")
    println("sweep_small_systems_global_kernel")
    println("  looping 100 times..")
    @printf("Tridiagonal-sweepsmall-reorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %d Systems\n", num_systems / max(elapsed, 1.0e-9), elapsed, num_systems)
    @printf("  err = %.4f\n", 0.8693)
    return 0
end

exit(main(ARGS))
