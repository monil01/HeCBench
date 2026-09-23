using CUDA
using Printf

const TPB = 1024
const TPB32 = Int32(1024)
const RAND_MAX_C = Float32(2147483647)

function rand_float()
    return Float32(ccall(:rand, Cint, ())) / RAND_MAX_C * Float32(2) - Float32(1)
end

function make_random_float(n::Integer)
    out = Vector{Float32}(undef, n)
    @inbounds for i in eachindex(out)
        out[i] = rand_float()
    end
    return out
end

function groupnorm_forward_ref!(x, weight, bias, out, mean, rstd, B, C, img_size, n_groups)
    group_size = C ÷ n_groups
    group_pixels = img_size * group_size
    eps = Float32(1f-5)
    @inbounds for b in 0:(B - 1), g in 0:(n_groups - 1)
        block_idx = b * n_groups + g
        base = block_idx * group_pixels
        wbase = g * group_size
        sumv = Float32(0)
        sum2 = Float32(0)
        for i in 1:group_pixels
            val = x[base + i]
            sumv += val
            sum2 += val * val
        end
        m = sumv / Float32(group_pixels)
        var = sum2 / Float32(group_pixels) - m * m
        s = inv(sqrt(var + eps))
        mean[block_idx + 1] = m
        rstd[block_idx + 1] = s
        for i in 0:(group_pixels - 1)
            c = i ÷ img_size
            n = s * (x[base + i + 1] - m)
            out[base + i + 1] = n * weight[wbase + c + 1] + bias[wbase + c + 1]
        end
    end
end

function groupnorm_backward_ref!(dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, n_groups)
    group_size = C ÷ n_groups
    group_pixels = img_size * group_size
    @inbounds for b in 0:(B - 1), g in 0:(n_groups - 1)
        block_idx = b * n_groups + g
        base = block_idx * group_pixels
        wbase = g * group_size
        m = mean[block_idx + 1]
        rs = rstd[block_idx + 1]
        w_dout_sum = Float32(0)
        w_dout_norm_sum = Float32(0)
        for i in 0:(group_pixels - 1)
            c = i ÷ img_size
            cur = weight[wbase + c + 1] * dout[base + i + 1]
            w_dout_sum += cur
            norm = (x[base + i + 1] - m) * rs
            w_dout_norm_sum += cur * norm
        end
        w_dout_block = w_dout_sum / Float32(group_pixels)
        w_dout_norm_block = w_dout_norm_sum / Float32(group_pixels)
        for i in 0:(group_pixels - 1)
            c = i ÷ img_size
            dout_val = dout[base + i + 1]
            norm = (x[base + i + 1] - m) * rs
            w_dout = weight[wbase + c + 1] * dout_val
            dx[base + i + 1] = (w_dout - w_dout_block - norm * w_dout_norm_block) * rs
        end
        for c in 0:(group_size - 1)
            dw = Float32(0)
            db = Float32(0)
            ch_base = base + c * img_size
            for i in 1:img_size
                dout_val = dout[ch_base + i]
                db += dout_val
                norm = (x[ch_base + i] - m) * rs
                dw += dout_val * norm
            end
            dweight[wbase + c + 1] += dw
            dbias[wbase + c + 1] += db
        end
    end
end

function reduce_shared!(shared, value)
    tid = threadIdx().x
    @inbounds shared[tid] = value
    sync_threads()
    offset = blockDim().x ÷ Int32(2)
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        sync_threads()
        offset ÷= Int32(2)
    end
    return shared[1]
end

function groupnorm_forward_kernel!(x, weight, bias, out, mean, rstd,
                                   img_size::Int32, group_size::Int32, n_groups::Int32)
    tid0 = threadIdx().x - Int32(1)
    block0 = blockIdx().x - Int32(1)
    group_pixels = img_size * group_size
    g = block0 % n_groups
    base = block0 * group_pixels
    wbase = g * group_size
    shared = CUDA.@cuStaticSharedMem(Float32, TPB)

    if tid0 == 0
        sumv = Float32(0)
        sum2 = Float32(0)
        @inbounds for i in Int32(0):(group_pixels - Int32(1))
            val = x[base + i + Int32(1)]
            sumv += val
            sum2 += val * val
        end
        inv_group_pixels = Float32(1) / Float32(group_pixels)
        m = sumv * inv_group_pixels
        var = sum2 * inv_group_pixels - m * m
        s = inv(sqrt(var + Float32(1f-5)))
        @inbounds begin
            mean[block0 + Int32(1)] = m
            rstd[block0 + Int32(1)] = s
        end
    end
    sync_threads()
    m = mean[block0 + Int32(1)]
    s = rstd[block0 + Int32(1)]
    i = tid0
    @inbounds while i < group_pixels
        c = i ÷ img_size
        n = s * (x[base + i + Int32(1)] - m)
        out[base + i + Int32(1)] = n * weight[wbase + c + Int32(1)] + bias[wbase + c + Int32(1)]
        i += blockDim().x
    end
    return
end

function groupnorm_backward_kernel!(dout, x, mean, rstd, weight, dx, dweight, dbias,
                                    img_size::Int32, group_size::Int32, n_groups::Int32)
    tid0 = threadIdx().x - Int32(1)
    block0 = blockIdx().x - Int32(1)
    group_pixels = img_size * group_size
    g = block0 % n_groups
    base = block0 * group_pixels
    wbase = g * group_size
    shared = CUDA.@cuStaticSharedMem(Float32, TPB)

    m = mean[block0 + Int32(1)]
    rs = rstd[block0 + Int32(1)]
    wd = Float32(0)
    wdn = Float32(0)
    i = tid0
    @inbounds while i < group_pixels
        c = i ÷ img_size
        cur = weight[wbase + c + Int32(1)] * dout[base + i + Int32(1)]
        wd += cur
        norm = (x[base + i + Int32(1)] - m) * rs
        wdn += cur * norm
        i += blockDim().x
    end
    inv_group_pixels = Float32(1) / Float32(group_pixels)
    wd_block = reduce_shared!(shared, wd) * inv_group_pixels
    wdn_block = reduce_shared!(shared, wdn) * inv_group_pixels

    i = tid0
    @inbounds while i < group_pixels
        c = i ÷ img_size
        dout_val = dout[base + i + Int32(1)]
        norm = (x[base + i + Int32(1)] - m) * rs
        w_dout = weight[wbase + c + Int32(1)] * dout_val
        dx[base + i + Int32(1)] = (w_dout - wd_block - norm * wdn_block) * rs
        i += blockDim().x
    end

    c = Int32(0)
    while c < group_size
        dw = Float32(0)
        db = Float32(0)
        i = tid0
        ch_base = base + c * img_size
        @inbounds while i < img_size
            dout_val = dout[ch_base + i + Int32(1)]
            db += dout_val
            norm = (x[ch_base + i + Int32(1)] - m) * rs
            dw += dout_val * norm
            i += blockDim().x
        end
        dw_sum = reduce_shared!(shared, dw)
        db_sum = reduce_shared!(shared, db)
        if tid0 == 0
            @inbounds begin
                CUDA.@atomic dweight[wbase + c + Int32(1)] += dw_sum
                CUDA.@atomic dbias[wbase + c + Int32(1)] += db_sum
            end
        end
        c += Int32(1)
    end
    return
end

function block_size(img_size, group_size)
    return max(min(TPB, img_size * group_size), 32)
end

function validate_result(device_result, cpu_reference, name, tolerance; n_print=5)
    out_gpu = Array(device_result)
    nfaults = 0
    @inbounds for i in eachindex(cpu_reference)
        if abs(cpu_reference[i] - out_gpu[i]) > tolerance && isfinite(cpu_reference[i])
            @printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i - 1, cpu_reference[i], out_gpu[i])
            nfaults += 1
            nfaults >= max(10, n_print) && break
        end
    end
    println(nfaults == 0 ? "PASS" : "FAIL")
    return nfaults == 0
end

function benchmark_kernel(f, repeat)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        f()
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1e-3 / repeat
end

function run_case(B, C, H, W, n_groups, repeat)
    img_size = H * W
    group_size = C ÷ n_groups
    n_blocks = B * n_groups
    bs_fwd = block_size(img_size, group_size)
    bs_bwd = max(min(TPB, img_size * group_size), 32 * group_size)

    ccall(:srand, Cvoid, (Cuint,), UInt32(0))
    x = make_random_float(B * C * img_size)
    weight = make_random_float(C)
    bias = make_random_float(C)
    dout = make_random_float(B * C * img_size)

    out = Vector{Float32}(undef, B * C * img_size)
    dx = zeros(Float32, B * C * img_size)
    dweight = zeros(Float32, C)
    dbias = zeros(Float32, C)
    mean = Vector{Float32}(undef, B * n_groups)
    rstd = Vector{Float32}(undef, B * n_groups)

    d_x = CuArray(x)
    d_weight = CuArray(weight)
    d_bias = CuArray(bias)
    d_out = CUDA.zeros(Float32, B * C * img_size)
    d_mean = CUDA.zeros(Float32, B * n_groups)
    d_rstd = CUDA.zeros(Float32, B * n_groups)
    d_dout = CuArray(dout)
    d_dx = CUDA.zeros(Float32, B * C * img_size)
    d_dweight = CUDA.zeros(Float32, C)
    d_dbias = CUDA.zeros(Float32, C)

    println("Checking forward pass")
    groupnorm_forward_ref!(x, weight, bias, out, mean, rstd, B, C, img_size, n_groups)
    @cuda threads=bs_fwd blocks=n_blocks groupnorm_forward_kernel!(
        d_x, d_weight, d_bias, d_out, d_mean, d_rstd, Int32(img_size), Int32(group_size), Int32(n_groups))
    CUDA.synchronize()
    validate_result(d_out, out, "out", Float32(1f-2))

    println("Checking forward2 pass")
    @cuda threads=bs_fwd blocks=n_blocks groupnorm_forward_kernel!(
        d_x, d_weight, d_bias, d_out, d_mean, d_rstd, Int32(img_size), Int32(group_size), Int32(n_groups))
    CUDA.synchronize()
    validate_result(d_out, out, "out", Float32(1f-2))

    println("Checking backward pass")
    groupnorm_backward_ref!(dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, n_groups)
    fill!(d_dweight, 0)
    fill!(d_dbias, 0)
    @cuda threads=bs_bwd blocks=n_blocks groupnorm_backward_kernel!(
        d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
        Int32(img_size), Int32(group_size), Int32(n_groups))
    CUDA.synchronize()
    println("Checking dbias")
    validate_result(d_dbias, dbias, "dbias", Float32(1f-2))
    println("Checking dweight")
    validate_result(d_dweight, dweight, "dweight", Float32(1f-2))
    println("Checking dx")
    validate_result(d_dx, dx, "dx", Float32(1.0))
    println()
    println("─────────────────────────────────────────────────────")

    println("Forward pass benchmarks")
    elapsed = benchmark_kernel(repeat) do
        @cuda threads=bs_fwd blocks=n_blocks groupnorm_forward_kernel!(
            d_x, d_weight, d_bias, d_out, d_mean, d_rstd, Int32(img_size), Int32(group_size), Int32(n_groups))
    end
    @printf("time %.4f us\n", elapsed)

    println("Forward2 pass benchmarks")
    elapsed = benchmark_kernel(repeat) do
        @cuda threads=bs_fwd blocks=n_blocks groupnorm_forward_kernel!(
            d_x, d_weight, d_bias, d_out, d_mean, d_rstd, Int32(img_size), Int32(group_size), Int32(n_groups))
    end
    @printf("time %.4f us\n", elapsed)

    println("Backward pass benchmarks")
    elapsed = benchmark_kernel(repeat) do
        @cuda threads=bs_bwd blocks=n_blocks groupnorm_backward_kernel!(
            d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
            Int32(img_size), Int32(group_size), Int32(n_groups))
    end
    @printf("time %.4f us\n", elapsed)
end

function main()
    if length(ARGS) != 6
        println("Usage: main.jl <batch size> <number of channels> <height> <width> <number of groups> <repeat>")
        exit(1)
    end
    vals = parse.(Int, ARGS)
    run_case(vals...)
end

main()
