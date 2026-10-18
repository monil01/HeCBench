using CUDA
using Printf

const LRN_SIZE = Int32(5)
const ALPHA = 0.000122f0
const BETA = 0.75f0
const KVAL = 1.0f0
const RAND_MAX_F32 = Float32(typemax(Cint))

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand_float()
    return Float32(ccall(:rand, Cint, ())) / RAND_MAX_F32
end

function lrn_fwd_kernel!(src, dst, total::Int64, cdim::Int32, ddim::Int32, hdim::Int32,
                         wdim::Int32, stride_mb::Int64, size::Int32,
                         alpha::Float32, kval::Float32)
    idx0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    grid_stride = Int64(blockDim().x) * Int64(gridDim().x)
    half_size = (size - Int32(1)) ÷ Int32(2)
    plane = Int64(hdim) * Int64(wdim)
    cube = Int64(ddim) * plane
    sample = Int64(cdim) * cube
    while idx0 < total
        mb = idx0 ÷ sample
        rem = idx0 % sample
        oc = Int32(rem ÷ cube)
        rem %= cube
        od = Int32(rem ÷ plane)
        rem %= plane
        oh = Int32(rem ÷ Int64(wdim))
        ow = Int32(rem % Int64(wdim))

        c_st = max(oc - half_size, Int32(0))
        c_en = min(oc + half_size + Int32(1), cdim)
        sum = 0.0f0
        for c in c_st:(c_en - Int32(1))
            off = mb * stride_mb + Int64(c) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
            s = @inbounds src[off]
            sum += s * s
        end
        omega = kval + alpha * sum / Float32(size)
        off = idx0 + 1
        s = @inbounds src[off]
        @inbounds dst[off] = s * sqrt(1.0f0 / (sqrt(omega) * omega))
        idx0 += grid_stride
    end
    return
end

function lrn_bwd_kernel!(src, dst, diff_src, total::Int64, cdim::Int32, ddim::Int32,
                         hdim::Int32, wdim::Int32, stride_mb::Int64,
                         size::Int32, alpha::Float32, beta::Float32, kval::Float32)
    idx0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    grid_stride = Int64(blockDim().x) * Int64(gridDim().x)
    half_size = (size - Int32(1)) ÷ Int32(2)
    plane = Int64(hdim) * Int64(wdim)
    cube = Int64(ddim) * plane
    sample = Int64(cdim) * cube
    while idx0 < total
        mb = idx0 ÷ sample
        rem = idx0 % sample
        oc = Int32(rem ÷ cube)
        rem %= cube
        od = Int32(rem ÷ plane)
        rem %= plane
        oh = Int32(rem ÷ Int64(wdim))
        ow = Int32(rem % Int64(wdim))

        c_st = max(oc - half_size, Int32(0))
        c_en = min(oc + half_size + Int32(1), cdim)
        aval = 0.0f0
        bval = 0.0f0
        for c in c_st:(c_en - Int32(1))
            omega_sum = 0.0f0
            cc_st = max(c - half_size, Int32(0))
            cc_en = min(c + half_size + Int32(1), cdim)
            for cc in cc_st:(cc_en - Int32(1))
                omega_off = mb * stride_mb + Int64(cc) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
                s = @inbounds src[omega_off]
                omega_sum += s * s
            end
            omega = kval + alpha * omega_sum / Float32(size)
            omega_in_beta = sqrt(1.0f0 / (sqrt(omega) * omega))
            off = mb * stride_mb + Int64(c) * cube + Int64(od) * plane + Int64(oh) * Int64(wdim) + Int64(ow) + 1
            dst_val = @inbounds dst[off]
            tmp = omega_in_beta * dst_val
            if c == oc
                aval = tmp
            end
            src_val = @inbounds src[off]
            bval += src_val * tmp / omega
        end
        off = idx0 + 1
        src_val = @inbounds src[off]
        bval *= 2.0f0 * alpha * beta * src_val / Float32(size)
        @inbounds diff_src[off] = aval - bval
        idx0 += grid_stride
    end
    return
end

@inline function lrn_offset(mb, c, d, h, w, cdim, ddim, hdim, wdim)
    return (((mb * cdim + c) * ddim + d) * hdim + h) * wdim + w + 1
end

function cpu_forward_checksum(src, n, cdim, ddim, hdim, wdim)
    total = length(src)
    dst_sum = 0.0
    half_size = (Int(LRN_SIZE) - 1) ÷ 2
    for mb in 0:(n - 1), c in 0:(cdim - 1), d in 0:(ddim - 1), h in 0:(hdim - 1), w in 0:(wdim - 1)
        ssum = 0.0f0
        for cc in max(c - half_size, 0):min(c + half_size, cdim - 1)
            s = src[lrn_offset(mb, cc, d, h, w, cdim, ddim, hdim, wdim)]
            ssum += s * s
        end
        omega = KVAL + ALPHA * ssum / Float32(LRN_SIZE)
        s = src[lrn_offset(mb, c, d, h, w, cdim, ddim, hdim, wdim)]
        dst_sum += Float64(s * sqrt(1.0f0 / (sqrt(omega) * omega)))
    end
    return dst_sum / total
end

function cpu_backward_checksum(src, dst, n, cdim, ddim, hdim, wdim)
    total = length(src)
    out_sum = 0.0
    half_size = (Int(LRN_SIZE) - 1) ÷ 2
    for mb in 0:(n - 1), oc in 0:(cdim - 1), d in 0:(ddim - 1), h in 0:(hdim - 1), w in 0:(wdim - 1)
        aval = 0.0f0
        bval = 0.0f0
        for c in max(oc - half_size, 0):min(oc + half_size, cdim - 1)
            omega_sum = 0.0f0
            for cc in max(c - half_size, 0):min(c + half_size, cdim - 1)
                s = src[lrn_offset(mb, cc, d, h, w, cdim, ddim, hdim, wdim)]
                omega_sum += s * s
            end
            omega = KVAL + ALPHA * omega_sum / Float32(LRN_SIZE)
            omega_in_beta = sqrt(1.0f0 / (sqrt(omega) * omega))
            off = lrn_offset(mb, c, d, h, w, cdim, ddim, hdim, wdim)
            tmp = omega_in_beta * dst[off]
            if c == oc
                aval = tmp
            end
            bval += src[off] * tmp / omega
        end
        off = lrn_offset(mb, oc, d, h, w, cdim, ddim, hdim, wdim)
        bval *= 2.0f0 * ALPHA * BETA * src[off] / Float32(LRN_SIZE)
        out_sum += Float64(aval - bval)
    end
    return out_sum / total
end

function launch_forward!(d_src, d_dst, total, n, cdim, ddim, hdim, wdim, wg_size, repeat)
    blocks = cld(total, wg_size)
    stride_mb = Int64(cdim) * Int64(ddim) * Int64(hdim) * Int64(wdim)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=wg_size blocks=blocks lrn_fwd_kernel!(
            d_src, d_dst, Int64(total), Int32(cdim), Int32(ddim), Int32(hdim),
            Int32(wdim), stride_mb, LRN_SIZE, ALPHA, KVAL)
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1.0e-9
end

function launch_backward!(d_src, d_dst, d_diff, total, n, cdim, ddim, hdim, wdim, wg_size, repeat)
    blocks = cld(total, wg_size)
    stride_mb = Int64(cdim) * Int64(ddim) * Int64(hdim) * Int64(wdim)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=wg_size blocks=blocks lrn_bwd_kernel!(
            d_src, d_dst, d_diff, Int64(total), Int32(cdim), Int32(ddim), Int32(hdim),
            Int32(wdim), stride_mb, LRN_SIZE, ALPHA, BETA, KVAL)
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1.0e-9
end

function run_case(repeat, n, cdim, ddim, hdim, wdim)
    total = n * cdim * ddim * hdim * wdim
    libc_srand(123)
    src = Vector{Float32}(undef, total)
    for i in 1:total
        src[i] = libc_rand_float()
    end
    dst = zeros(Float32, total)
    diff_src = copy(src)

    d_src = CuArray(src)
    d_dst = CuArray(dst)
    d_diff = CuArray(diff_src)

    println("Sweep the work-group sizes from 64 to 512")
    fwd_time = 0.0
    for wg_size in (64, 128, 256, 512)
        fwd_time = launch_forward!(d_src, d_dst, total, n, cdim, ddim, hdim, wdim, wg_size, repeat)
        @printf("Average execution time of lrn_fwd_kernel: %.6f sec \n", fwd_time / repeat)
        data_in_gb = (2 * total * sizeof(Float32)) / 1.0e9
        @printf("Kernel bandwidth: %.6f GB/s \n", data_in_gb * repeat / fwd_time)
    end
    fwd_checksum = sum(Float64, Array(d_dst)) / total
    @printf("Checksum: %.6f\n", fwd_checksum)

    ref_fwd = cpu_forward_checksum(src, n, cdim, ddim, hdim, wdim)
    if abs(fwd_checksum - ref_fwd) <= 1.0e-5
        println("PASS")
    else
        @printf("FAIL forward checksum reference %.9f\n", ref_fwd)
        return false
    end

    d_dst_bwd = CuArray(src)
    d_diff_bwd = CuArray(diff_src)
    println("Sweep the work-group sizes from 64 to 512")
    bwd_time = 0.0
    for wg_size in (64, 128, 256, 512)
        bwd_time = launch_backward!(d_src, d_dst_bwd, d_diff_bwd, total, n, cdim, ddim, hdim, wdim, wg_size, repeat)
        @printf("Average execution time of lrn_bwd_kernel: %.6f sec \n", bwd_time / repeat)
        data_in_gb = (3 * total * sizeof(Float32)) / 1.0e9
        @printf("Kernel bandwidth: %.6f GB/s \n", data_in_gb * repeat / bwd_time)
    end
    bwd_checksum = sum(Float64, Array(d_diff_bwd)) / total
    @printf("Checksum: %.6f\n", bwd_checksum)

    ref_bwd = cpu_backward_checksum(src, src, n, cdim, ddim, hdim, wdim)
    if abs(bwd_checksum - ref_bwd) <= 1.0e-5
        println("PASS")
        return true
    end
    @printf("FAIL backward checksum reference %.9f\n", ref_bwd)
    return false
end

function main(args)
    if !(length(args) == 1 || length(args) == 6)
        println("Usage: main.jl <repeat> [N C D H W]")
        return 1
    end
    repeat = parse(Int, args[1])
    if length(args) == 6
        dims = parse.(Int, args[2:6])
    else
        dims = [6, 150, 100, 160, 160]
    end
    all(>(0), dims) || return 1
    return run_case(repeat, dims...) ? 0 : 1
end

exit(main(ARGS))
