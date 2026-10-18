using CUDA
using Printf

# Julia port of bsearch-cuda benchmark
# Only kernel BS1 (basic while-loop binary search) is ported;
# other kernel variants (BS2..BS4) exercise the same interface.

function bs_kernel!(a, z, r, zSize::Int64, n::Int64)
    i = Int64((blockIdx().x - 1)) * Int64(blockDim().x) + Int64(threadIdx().x)
    if i > zSize
        return
    end
    @inbounds begin
        zi = z[i]
        low  = Int64(0)   # 0-based logical index, like the C code
        high = n          # exclusive upper bound (= aSize - 1)
        while high - low > 1
            mid = low + (high - low) ÷ 2
            # a is 1-based Julia array: element at logical index k is a[k+1]
            if zi < a[mid + 1]
                high = mid
            else
                low = mid
            end
        end
        r[i] = low
    end
    return
end

function cpu_ref(a::Vector{Float32}, z::Vector{Float32}, n::Int64)
    zSize = length(z)
    r = Vector{Int64}(undef, zSize)
    for i in 1:zSize
        zi = z[i]
        low = Int64(0); high = n
        while high - low > 1
            mid = low + (high - low) ÷ 2
            if zi < a[mid + 1]
                high = mid
            else
                low = mid
            end
        end
        r[i] = low
    end
    return r
end

function main()
    if length(ARGS) < 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    numElem = parse(Int64, ARGS[1])
    repeat_n = parse(Int, ARGS[2])

    aSize = numElem
    zSize = 2 * aSize
    N = aSize - 1

    a = Vector{Float32}(undef, aSize)
    for i in 1:aSize
        a[i] = Float32(i - 1)   # strictly ascending: 0, 1, ..., aSize-1
    end

    # Deterministic values in [0, N) via LCG
    z = Vector{Float32}(undef, zSize)
    state = UInt64(2)
    for i in 1:zSize
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        z[i] = Float32((state >> 11) % UInt64(N))
    end

    d_a = CuArray(a)
    d_z = CuArray(z)
    d_r = CUDA.zeros(Int64, zSize)

    threads = 256
    blocks = cld(zSize, threads)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks bs_kernel!(d_a, d_z, d_r, Int64(zSize), Int64(N))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / repeat_n
    @printf("Average kernel execution time (bs1) %f (s)\n", elapsed_s)

    r_gpu = Array(d_r)

    # Verify against structural invariant: a[r] <= z < a[r+1]
    ok = true
    @inbounds for i in 1:zSize
        idx = r_gpu[i]
        if !(idx + 1 < aSize && a[idx + 1] <= z[i] && z[i] < a[idx + 2])
            ok = false
            println("mismatch at $i: r=$idx z=$(z[i])")
            break
        end
    end

    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
