using CUDA
using Printf

const GRID_SIZE = 12 * 7 * 8 * 9 * 10

valid(x::Int32) = x > 0

function binary_scan_kernel!(g_odata, g_idata, n::Int32)
    tid0 = threadIdx().x - Int32(1)
    if tid0 < n
        total = Int32(0)
        k = Int32(0)
        while k < tid0
            total += ifelse(g_idata[k + Int32(1)] > 0, Int32(1), Int32(0))
            k += Int32(1)
        end
        g_odata[tid0 + Int32(1)] = total
    end
    return
end

function c_rand_mod(n::Int)
    return Int(ccall(:rand, Cint, ())) % n
end

function run_bscan(n::Int, repeat::Int)
    h_in = Vector{Int32}(undef, n)
    h_out = Vector{Int32}(undef, n)
    ref_out = Vector{Int32}(undef, n)
    d_in = CuArray{Int32}(undef, n)
    d_out = CuArray{Int32}(undef, n)

    ok = true
    elapsed_ns = 0.0
    valid_count = 0

    for _ in 1:repeat
        for i in 1:n
            h_in[i] = Int32(c_rand_mod(n) - n ÷ 2)
            valid_count += valid(h_in[i]) ? 1 : 0
        end
        copyto!(d_in, h_in)

        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=n blocks=GRID_SIZE binary_scan_kernel!(d_out, d_in, Int32(n))
        CUDA.synchronize()
        elapsed_ns += Float64(time_ns() - t0)

        copyto!(h_out, d_out)
        ref_out[1] = Int32(0)
        ok &= h_out[1] == ref_out[1]
        for i in 2:n
            ref_out[i] = ref_out[i - 1] + (h_in[i - 1] > 0 ? Int32(1) : Int32(0))
            ok &= ref_out[i] == h_out[i]
        end
        ok || break
    end

    @printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
            n, valid_count / (n * repeat), ok ? "PASS" : "FAIL")
    if ok
        @printf("Average execution time: %f (us)\n", (elapsed_ns * 1e-3) / repeat)
        @printf("Billion elements per second: %f\n\n",
                GRID_SIZE * n * repeat / elapsed_ns)
    end
    return ok
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    ccall(:srand, Cvoid, (Cuint,), Cuint(123))
    ok = true
    for n in (32, 64, 128, 256, 512, 1024)
        ok &= run_bscan(n, repeat)
    end
    ok || exit(1)
end

main()
