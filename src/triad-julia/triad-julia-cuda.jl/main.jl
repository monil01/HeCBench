using CUDA
using Printf

# Julia port of triad-cuda benchmark (simplified)
# Kernel: C = A + s*B

function triad_kernel!(A, B, C, s::Float32)
    gid = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    @inbounds if gid <= length(C)
        C[gid] = A[gid] + s * B[gid]
    end
    return
end

function main()
    # Skip flag-style args (--passes N -v etc.); accept the first positional
    # numeric argument as `passes` for compatibility with the CUDA CLI.
    numeric = filter(a -> !startswith(a, "-"), ARGS)
    passes_from_flag = ""
    for i in 1:length(ARGS)-1
        if ARGS[i] in ("--passes","-p") && !startswith(ARGS[i+1], "-")
            passes_from_flag = ARGS[i+1]
        end
    end
    if passes_from_flag != ""
        pushfirst!(numeric, passes_from_flag)
    end
    if length(numeric) < 1
        println("Usage: main.jl <passes>  (or --passes N)")
        return 1
    end
    n_passes = parse(Int, numeric[1])

    # 8M floats total, but split into two halves and pipelined in CUDA version;
    # here we run a single fixed-size pass and verify halves match.
    numMaxFloats = 1024 * 16384 ÷ sizeof(Float32)   # 4,194,304 floats = 16 MB
    halfNumFloats = numMaxFloats ÷ 2

    scalar = 1.75f0

    # Deterministic init: fill both halves identically
    h_mem = Vector{Float32}(undef, numMaxFloats)
    state = Ref(UInt64(8650341))
    function next_f32()
        state[] = state[] * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(((state[] >> 11) & UInt64(0xFFFFFF))) / Float32(1 << 24) * 10.0f0
    end
    for j in 1:halfNumFloats
        v = next_f32()
        h_mem[j] = v
        h_mem[j + halfNumFloats] = v
    end

    # Use just one buffer set; time triad kernel over the full array
    d_A = CuArray(h_mem)
    d_B = CuArray(h_mem)
    d_C = CUDA.zeros(Float32, numMaxFloats)

    threads = 128
    blocks  = cld(numMaxFloats, threads)

    # Warmup + timing
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:n_passes
        @cuda threads=threads blocks=blocks triad_kernel!(d_A, d_B, d_C, scalar)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - t0) * 1e-9
    gflops = (2.0 * numMaxFloats * n_passes) / (elapsed * 1e9)
    bdwth  = (Float64(sizeof(Float32)) * 3.0 * numMaxFloats * n_passes) / (elapsed * 1e9)
    @printf("Average TriadFlops %f GFLOPS/s\n", gflops)
    @printf("Average TriadBdwth %f GB/s\n", bdwth)

    C_gpu = Array(d_C)

    # Verify: for identical halves, both should equal A + s*B
    ok = true
    for j in 1:halfNumFloats
        ref = h_mem[j] + scalar * h_mem[j]
        if abs(C_gpu[j] - ref) > 1e-3 || C_gpu[j] != C_gpu[j + halfNumFloats]
            ok = false
            break
        end
    end

    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
