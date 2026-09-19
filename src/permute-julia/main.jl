using CUDA
using Printf
using Random

function permute_kernel!(q, k, v, inp, total::Int32, tdim::Int32, nh::Int32, d::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 < total
        c = nh * d
        b = idx0 ÷ (c * tdim)
        rest = idx0 % (c * tdim)
        nh_ = rest ÷ (tdim * d)
        rest = rest % (tdim * d)
        n = rest ÷ d
        d_ = rest % d
        inp_idx = b * tdim * Int32(3) * c + n * Int32(3) * c + nh_ * d + d_ + Int32(1)
        out_idx = idx0 + Int32(1)
        @inbounds begin
            q[out_idx] = inp[inp_idx]
            k[out_idx] = inp[inp_idx + c]
            v[out_idx] = inp[inp_idx + Int32(2) * c]
        end
    end
    return
end

function permute_cpu(inp, bsz, tdim, c, nh)
    d = c ÷ nh
    total = bsz * tdim * c
    q = Vector{Float32}(undef, total)
    k = Vector{Float32}(undef, total)
    v = Vector{Float32}(undef, total)
    i = 1
    for b in 0:(bsz - 1), n in 0:(nh - 1), t in 0:(tdim - 1), cc in (n*d):((n+1)*d - 1)
        base = b * tdim * 3 * c + t * 3 * c + cc + 1
        q[i] = inp[base]
        k[i] = inp[base + c]
        v[i] = inp[base + 2c]
        i += 1
    end
    q, k, v
end

function run_permute!(out, inp, bsz, tdim, c, nh, block_size)
    d = c ÷ nh
    total = bsz * tdim * c
    q = view(out, 1:total)
    k = view(out, (total + 1):(2total))
    v = view(out, (2total + 1):(3total))
    blocks = cld(total, block_size)
    @cuda threads=block_size blocks=blocks permute_kernel!(
        q, k, v, inp, Int32(total), Int32(tdim), Int32(nh), Int32(d))
    CUDA.synchronize()
end

function validate(actual, ref, name)
    h = Array(actual)
    for i in eachindex(ref)
        if abs(h[i] - ref[i]) > 1.0f-6
            @printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i - 1, ref[i], h[i])
            return false
        end
    end
    return true
end

function benchmark(repeat_times, out, inp, bsz, tdim, c, nh, block_size)
    elapsed = 0.0
    for _ in 1:repeat_times
        t0 = time_ns()
        run_permute!(out, inp, bsz, tdim, c, nh, block_size)
        elapsed += (time_ns() - t0) * 1.0e-6
    end
    elapsed / repeat_times
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <batch size> <repeat>")
        return 1
    end
    bsz = parse(Int, args[1])
    repeat_times = parse(Int, args[2])
    tdim = 1024
    c = 768
    nh = 12
    total = bsz * tdim * c

    rng = MersenneTwister(123)
    inp = rand(rng, Float32, 3total) .* 2.0f0 .- 1.0f0
    q_ref, k_ref, v_ref = permute_cpu(inp, bsz, tdim, c, nh)
    d_inp = CuArray(inp)
    d_out = CUDA.zeros(Float32, 3total)

    block_sizes = (32, 64, 128, 256, 512, 1024)
    for block_size in block_sizes
        @printf("Checking block size %d.\n", block_size)
        run_permute!(d_out, d_inp, bsz, tdim, c, nh, block_size)
        total_range = total
        ok = validate(view(d_out, 1:total_range), q_ref, "q")
        ok &= validate(view(d_out, (total + 1):(2total)), k_ref, "k")
        ok &= validate(view(d_out, (2total + 1):(3total)), v_ref, "v")
        ok || return 1
    end
    println("All results match. Starting benchmarks.")
    println()

    for block_size in block_sizes
        elapsed = benchmark(repeat_times, d_out, d_inp, bsz, tdim, c, nh, block_size)
        @printf("block_size %4d | time %f ms\n", block_size, elapsed)
    end
    return 0
end

exit(main(ARGS))
