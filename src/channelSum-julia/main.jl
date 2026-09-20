using CUDA
using Printf
using Random

const NUM_THREADS = 256

function channel_sum_nhwc_kernel!(n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = blockIdx().x - Int32(1)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    inner = n * hxw
    sval = Int32(0)
    qval = Int32(0)
    i0 = tid0
    while i0 < inner
        @inbounds v = x[i0 * cnum + c0 + Int32(1)]
        sval += v
        qval += v * v
        i0 += blockDim().x
    end
    sm1 = CuStaticSharedArray(Int32, 256)
    sm2 = CuStaticSharedArray(Int32, 256)
    sm1[tid] = sval
    sm2[tid] = qval
    sync_threads()
    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds begin
            sum[c0 + Int32(1)] = sm1[Int32(1)]
            sumsq[c0 + Int32(1)] = sm2[Int32(1)]
        end
    end
    return
end

function channel_sum_nchw_kernel!(n::Int32, cnum::Int32, hxw::Int32, x, sum, sumsq)
    c0 = blockIdx().x - Int32(1)
    tid = threadIdx().x
    tid0 = tid - Int32(1)
    total = n * hxw
    sval = Int32(0)
    qval = Int32(0)
    i0 = tid0
    while i0 < total
        n0 = i0 ÷ hxw
        hw = i0 - n0 * hxw
        @inbounds v = x[(n0 * cnum + c0) * hxw + hw + Int32(1)]
        sval += v
        qval += v * v
        i0 += blockDim().x
    end
    sm1 = CuStaticSharedArray(Int32, 256)
    sm2 = CuStaticSharedArray(Int32, 256)
    sm1[tid] = sval
    sm2[tid] = qval
    sync_threads()
    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        if tid <= offset
            sm1[tid] += sm1[tid + offset]
            sm2[tid] += sm2[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds begin
            sum[c0 + Int32(1)] = sm1[Int32(1)]
            sumsq[c0 + Int32(1)] = sm2[Int32(1)]
        end
    end
    return
end

function ref_nhwc!(n, cnum, hxw, x, sum)
    for c0 in 0:cnum-1
        s = Int32(0)
        for i0 in 0:n*hxw-1
            s += x[i0 * cnum + c0 + 1]
        end
        sum[c0 + 1] = s
    end
end

function ref_nchw!(n, cnum, hxw, x, sum)
    for c0 in 0:cnum-1
        s = Int32(0)
        for n0 in 0:n-1, hw in 0:hxw-1
            s += x[(n0 * cnum + c0) * hxw + hw + 1]
        end
        sum[c0 + 1] = s
    end
end

function run_case(w::Int, h::Int, repeat::Int, n::Int, cnum::Int)
    @printf("\n(N=%d C=%d W=%d H=%d)\n", n, cnum, w, h)
    hxw = w * h
    numel = n * cnum * hxw
    rng = MersenneTwister(numel)
    x = Int32.(rand(rng, 0:255, numel))
    d_x = CuArray(x)
    d_sum = CUDA.zeros(Int32, cnum)
    d_sumsq = CUDA.zeros(Int32, cnum)
    h_sum = Vector{Int32}(undef, cnum)
    r_sum = Vector{Int32}(undef, cnum)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=NUM_THREADS blocks=cnum channel_sum_nhwc_kernel!(
            Int32(n), Int32(cnum), Int32(hxw), d_x, d_sum, d_sumsq)
    end
    CUDA.synchronize()
    copyto!(h_sum, d_sum)
    ref_nhwc!(n, cnum, hxw, x, r_sum)
    ok = all(abs.(h_sum .- r_sum) .<= 1)
    @printf("Average time of channel sum (nhwc): %f (ms)\n", (time_ns() - t0) * 1.0e-6 / repeat)
    @printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=NUM_THREADS blocks=cnum channel_sum_nchw_kernel!(
            Int32(n), Int32(cnum), Int32(hxw), d_x, d_sum, d_sumsq)
    end
    CUDA.synchronize()
    copyto!(h_sum, d_sum)
    ref_nchw!(n, cnum, hxw, x, r_sum)
    ok = all(abs.(h_sum .- r_sum) .<= 1)
    @printf("Average time of channel sum (nchw): %f (ms)\n", (time_ns() - t0) * 1.0e-6 / repeat)
    @printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL")
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <width> <height> <repeat>")
        return 1
    end
    w = parse(Int, args[1])
    h = parse(Int, args[2])
    repeat = parse(Int, args[3])
    for n in (1, 4, 16, 64)
        for cnum in (32, 128, 512)
            run_case(w, h, repeat, n, cnum)
        end
    end
    return 0
end

exit(main(ARGS))
