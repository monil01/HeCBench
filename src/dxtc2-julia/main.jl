using CUDA
using Printf
using StaticArrays

const ERROR_THRESHOLD = 0.02f0
const NUM_THREADS = 64
const FOURCC_DDS = UInt32(0x20534444)
const FOURCC_DXT1 = UInt32(0x31545844)

struct F3
    x::Float32
    y::Float32
    z::Float32
end

@inline f3zero() = F3(0.0f0, 0.0f0, 0.0f0)
@inline Base.:+(a::F3, b::F3) = F3(a.x + b.x, a.y + b.y, a.z + b.z)
@inline Base.:-(a::F3, b::F3) = F3(a.x - b.x, a.y - b.y, a.z - b.z)
@inline Base.:*(a::F3, b::F3) = F3(a.x * b.x, a.y * b.y, a.z * b.z)
@inline Base.:*(a::F3, b::Float32) = F3(a.x * b, a.y * b, a.z * b)
@inline Base.:*(b::Float32, a::F3) = a * b

function read_ppm4(path::AbstractString)
    bytes = read(path)
    pos = 1
    function next_token()
        while pos <= length(bytes)
            c = bytes[pos]
            if c == UInt8('#')
                while pos <= length(bytes) && bytes[pos] != UInt8('\n')
                    pos += 1
                end
            elseif c in (UInt8(' '), UInt8('\n'), UInt8('\r'), UInt8('\t'))
                pos += 1
            else
                break
            end
        end
        start = pos
        while pos <= length(bytes) && !(bytes[pos] in (UInt8(' '), UInt8('\n'), UInt8('\r'), UInt8('\t')))
            pos += 1
        end
        return String(bytes[start:pos-1])
    end
    magic = next_token()
    magic == "P6" || error("unsupported PPM magic $magic")
    w = parse(Int, next_token())
    h = parse(Int, next_token())
    maxval = parse(Int, next_token())
    maxval == 255 || error("unsupported PPM max value $maxval")
    while bytes[pos] in (UInt8(' '), UInt8('\n'), UInt8('\r'), UInt8('\t'))
        pos += 1
    end
    rgb = @view bytes[pos:end]
    length(rgb) >= w * h * 3 || error("truncated PPM")
    rgba = Vector{UInt8}(undef, w * h * 4)
    r = 1
    q = 1
    for _ in 1:(w*h)
        rgba[q] = rgb[r]
        rgba[q+1] = rgb[r+1]
        rgba[q+2] = rgb[r+2]
        rgba[q+3] = 0x00
        r += 3
        q += 4
    end
    return rgba, UInt32(w), UInt32(h)
end

function block_linear_image(rgba::Vector{UInt8}, w::UInt32, h::UInt32)
    blocks = Int((w ÷ 4) * (h ÷ 4))
    out = Vector{UInt32}(undef, blocks * 16)
    wi = Int(w)
    p = 1
    for by in 0:(Int(h ÷ 4)-1), bx in 0:(Int(w ÷ 4)-1), i in 0:15
        x = i & 3
        y = i >>> 2
        pix = ((by * 4 + y) * wi + bx * 4 + x) * 4 + 1
        out[p] = UInt32(rgba[pix]) | (UInt32(rgba[pix+1]) << 8) |
                 (UInt32(rgba[pix+2]) << 16) | (UInt32(rgba[pix+3]) << 24)
        p += 1
    end
    return out
end

function compute_permutations()
    permutations = Vector{UInt32}(undef, 1024)
    indices = zeros(Int32, 16)
    num = 1
    imax = 15
    for i in imax:-1:0
        for m in i:15
            indices[m+1] = 2
        end
        jmax = i == 0 ? 15 : 16
        for j in jmax:-1:i
            if j < 16
                indices[j+1] = 1
            end
            permutation = UInt32(0)
            for p in 0:15
                permutation |= UInt32(indices[p+1]) << UInt32(2p)
            end
            permutations[num] = permutation
            num += 1
        end
    end
    for _ in 1:9
        permutations[num] = 0x000aa555
        num += 1
    end
    fill!(indices, 0)
    for i in imax:-1:0
        for m in i:15
            indices[m+1] = 2
        end
        jmax = i == 0 ? 15 : 16
        for j in jmax:-1:i
            for m in j:15
                indices[m+1] = 3
            end
            kmax = j == 0 ? 15 : 16
            for k in kmax:-1:j
                if k < 16
                    indices[k+1] = 1
                end
                permutation = UInt32(0)
                has_three = false
                for p in 0:15
                    permutation |= UInt32(indices[p+1]) << UInt32(2p)
                    has_three |= indices[p+1] == 3
                end
                if has_three
                    permutations[num] = permutation
                    num += 1
                end
            end
        end
    end
    for _ in 1:49
        permutations[num] = 0x00aaff55
        num += 1
    end
    return permutations
end

@inline function first_eigen_vector(m)
    v = F3(1.0f0, 1.0f0, 1.0f0)
    for _ in 1:8
        x = v.x * m[1] + v.y * m[2] + v.z * m[3]
        y = v.x * m[2] + v.y * m[4] + v.z * m[5]
        z = v.x * m[3] + v.y * m[5] + v.z * m[6]
        mx = max(max(x, y), z)
        if !(mx > 0.0f0)
            return F3(1.0f0, 0.0f0, 0.0f0)
        end
        iv = 1.0f0 / mx
        v = F3(x * iv, y * iv, z * iv)
    end
    return v
end

@inline sat01(x::Float32) = ifelse(isfinite(x), min(max(x, 0.0f0), 1.0f0), 0.0f0)

@inline function round_and_expand(v::F3)
    rx = UInt16(floor(sat01(v.x) * 31.0f0 + 0.5f0))
    ry = UInt16(floor(sat01(v.y) * 63.0f0 + 0.5f0))
    rz = UInt16(floor(sat01(v.z) * 31.0f0 + 0.5f0))
    w = UInt16((rx << 11) | (ry << 5) | rz)
    return F3(Float32(rx) * 0.03227752766457f0,
              Float32(ry) * 0.01583151765563f0,
              Float32(rz) * 0.03227752766457f0), w
end

@inline function alpha4(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? 9.0f0 : b == 0x01 ? 0.0f0 : b == 0x02 ? 6.0f0 : 3.0f0
end

@inline function alpha3(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? 4.0f0 : b == 0x01 ? 0.0f0 : 2.0f0
end

@inline function prod4(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? Int32(0x090000) : b == 0x01 ? Int32(0x000900) :
           b == 0x02 ? Int32(0x040102) : Int32(0x010402)
end

@inline function prod3(bits::UInt32)
    b = bits & 0x03
    return b == 0x00 ? Int32(0x040000) : b == 0x01 ? Int32(0x000400) :
           b == 0x02 ? Int32(0x040101) : Int32(0x010401)
end

@inline function eval_permutation4(colors, permutation::UInt32, color_sum::F3)
    alphax_sum = f3zero()
    akku = Int32(0)
    for i in 0:15
        bits = permutation >> UInt32(2i)
        alphax_sum = alphax_sum + colors[i+1] * alpha4(bits)
        akku += prod4(bits)
    end
    alpha2_sum = Float32(akku >> 16)
    beta2_sum = Float32((akku >> 8) & 0xff)
    alphabeta_sum = Float32(akku & 0xff)
    betax_sum = color_sum * 9.0f0 - alphax_sum
    factor = 1.0f0 / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum)
    a, start = round_and_expand((alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor)
    b, stop = round_and_expand((betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor)
    e = a * a * alpha2_sum + b * b * beta2_sum +
        (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum) * 2.0f0
    return 0.111111111111f0 * (e.x + e.y + e.z), start, stop
end

@inline function eval_permutation3(colors, permutation::UInt32, color_sum::F3)
    alphax_sum = f3zero()
    akku = Int32(0)
    for i in 0:15
        bits = permutation >> UInt32(2i)
        alphax_sum = alphax_sum + colors[i+1] * alpha3(bits)
        akku += prod3(bits)
    end
    alpha2_sum = Float32(akku >> 16)
    beta2_sum = Float32((akku >> 8) & 0xff)
    alphabeta_sum = Float32(akku & 0xff)
    betax_sum = color_sum * 4.0f0 - alphax_sum
    factor = 1.0f0 / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum)
    a, start = round_and_expand((alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor)
    b, stop = round_and_expand((betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor)
    e = a * a * alpha2_sum + b * b * beta2_sum +
        (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum) * 2.0f0
    return 0.25f0 * (e.x + e.y + e.z), start, stop
end

function compress_kernel!(permutations, image, result, nblocks::Int32, block_offset::Int32)
    gid = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if gid > nblocks
        return
    end
    bid0 = block_offset + gid - Int32(1)
    colors = MVector{16,F3}(undef)
    sorted = MVector{16,F3}(undef)
    xrefs = MVector{16,Int32}(undef)
    dps = MVector{16,Float32}(undef)
    covariance = MVector{6,Float32}(undef)
    sums = f3zero()
    for i in 0:15
        c = image[Int(bid0) * 16 + i + 1]
        col = F3(Float32(c & 0xff) * (1.0f0 / 255.0f0),
                 Float32((c >> 8) & 0xff) * (1.0f0 / 255.0f0),
                 Float32((c >> 16) & 0xff) * (1.0f0 / 255.0f0))
        colors[i+1] = col
        sums = sums + col
    end
    for i in 1:6
        covariance[i] = 0.0f0
    end
    mean = sums * (1.0f0 / 16.0f0)
    for i in 1:16
        diff = colors[i] - mean
        covariance[1] += diff.x * diff.x
        covariance[2] += diff.x * diff.y
        covariance[3] += diff.x * diff.z
        covariance[4] += diff.y * diff.y
        covariance[5] += diff.y * diff.z
        covariance[6] += diff.z * diff.z
    end
    axis = first_eigen_vector(covariance)
    for i in 1:16
        dps[i] = colors[i].x * axis.x + colors[i].y * axis.y + colors[i].z * axis.z
    end
    for idx in 1:16
        rank = Int32(0)
        for i in 1:16
            rank += ifelse(dps[i] < dps[idx], Int32(1), Int32(0))
        end
        xrefs[idx] = rank
    end
    for i in 1:15
        for idx in 1:16
            if idx > i && xrefs[idx] == xrefs[i]
                xrefs[idx] += Int32(1)
            end
        end
    end
    for i in 1:16
        sorted[Int(xrefs[i]) + 1] = colors[i]
    end
    best_error = typemax(Float32)
    best_start = UInt16(0)
    best_end = UInt16(0)
    best_perm = UInt32(0)
    for pidx in 0:991
        err, start, stop = eval_permutation4(sorted, permutations[pidx+1], sums)
        if err < best_error
            best_error = err
            best_perm = permutations[pidx+1]
            best_start = start
            best_end = stop
        end
    end
    if best_start < best_end
        tmp = best_end
        best_end = best_start
        best_start = tmp
        best_perm ⊻= 0x55555555
    end
    for pidx in 0:159
        err, start, stop = eval_permutation3(sorted, permutations[pidx+1], sums)
        if err < best_error
            best_error = err
            best_perm = permutations[pidx+1]
            best_start = start
            best_end = stop
            if best_start > best_end
                tmp = best_end
                best_end = best_start
                best_start = tmp
                best_perm ⊻= ((~best_perm) >> 1) & 0x55555555
            end
        end
    end
    if best_start == best_end
        best_perm = 0x00000000
    end
    indices = UInt32(0)
    for i in 0:15
        ref = UInt32(xrefs[i+1])
        indices |= ((best_perm >> (2 * ref)) & 0x03) << UInt32(2i)
    end
    out = Int(bid0) * 2 + 1
    result[out] = (UInt32(best_end) << 16) | UInt32(best_start)
    result[out+1] = indices
    return
end

function write_u32_le!(io, x::UInt32)
    write(io, UInt8(x & 0xff), UInt8((x >> 8) & 0xff), UInt8((x >> 16) & 0xff), UInt8((x >> 24) & 0xff))
end

function write_dds(path::AbstractString, payload::Vector{UInt32}, w::UInt32, h::UInt32)
    compressed_size = UInt32((w ÷ 4) * (h ÷ 4) * 8)
    open(path, "w") do io
        write_u32_le!(io, FOURCC_DDS)
        write_u32_le!(io, UInt32(124))
        write_u32_le!(io, UInt32(0x00001007))
        write_u32_le!(io, h)
        write_u32_le!(io, w)
        write_u32_le!(io, compressed_size)
        write_u32_le!(io, UInt32(0))
        write_u32_le!(io, UInt32(0))
        for _ in 1:11
            write_u32_le!(io, UInt32(0))
        end
        write_u32_le!(io, UInt32(32))
        write_u32_le!(io, UInt32(0x00000004))
        write_u32_le!(io, FOURCC_DXT1)
        for _ in 1:5
            write_u32_le!(io, UInt32(0))
        end
        write_u32_le!(io, UInt32(0x00001000))
        for _ in 1:4
            write_u32_le!(io, UInt32(0))
        end
        for word in payload
            write_u32_le!(io, word)
        end
    end
end

@inline expand5(x::UInt32) = UInt32((x << 3) | (x >> 2))
@inline expand6(x::UInt32) = UInt32((x << 2) | (x >> 4))

function decompress_block(words::AbstractVector{UInt32}, block_idx::Int)
    word0 = words[2block_idx - 1]
    inds = words[2block_idx]
    col0 = word0 & 0xffff
    col1 = (word0 >> 16) & 0xffff
    p = Matrix{UInt32}(undef, 4, 3)
    p[1, 1] = expand5(col0 & 0x1f)
    p[1, 2] = expand6((col0 >> 5) & 0x3f)
    p[1, 3] = expand5((col0 >> 11) & 0x1f)
    p[2, 1] = expand5(col1 & 0x1f)
    p[2, 2] = expand6((col1 >> 5) & 0x3f)
    p[2, 3] = expand5((col1 >> 11) & 0x1f)
    if col0 > col1
        for c in 1:3
            p[3, c] = (2p[1, c] + p[2, c]) ÷ 3
            p[4, c] = (2p[2, c] + p[1, c]) ÷ 3
        end
    else
        for c in 1:3
            p[3, c] = (p[1, c] + p[2, c]) ÷ 2
            p[4, c] = 0
        end
    end
    colors = Matrix{Int32}(undef, 16, 3)
    for i in 0:15
        idx = Int((inds >> UInt32(2i)) & 0x03) + 1
        colors[i+1, 1] = Int32(p[idx, 3])
        colors[i+1, 2] = Int32(p[idx, 2])
        colors[i+1, 3] = Int32(p[idx, 1])
    end
    return colors
end

function compare_block(a::AbstractVector{UInt32}, b::AbstractVector{UInt32}, block_idx::Int)
    if a[2block_idx - 1] == b[2block_idx - 1] && a[2block_idx] == b[2block_idx]
        return 0
    end
    ca = decompress_block(a, block_idx)
    cb = decompress_block(b, block_idx)
    s = 0
    for i in 1:16, c in 1:3
        d = ca[i, c] - cb[i, c]
        s += d * d
    end
    return s
end

function read_dds_payload(path::AbstractString, nwords::Int)
    bytes = read(path)
    payload = @view bytes[129:end]
    length(payload) >= nwords * 4 || error("truncated DDS reference")
    out = Vector{UInt32}(undef, nwords)
    for i in 1:nwords
        p = 4i - 3
        out[i] = UInt32(payload[p]) | (UInt32(payload[p+1]) << 8) |
                 (UInt32(payload[p+2]) << 16) | (UInt32(payload[p+3]) << 24)
    end
    return out
end

function output_name(image_path::AbstractString)
    return replace(basename(image_path), r"ppm$" => "dds")
end

function run_case(image_path::String, reference_path::String, num_iterations::Int)
    data, w, h = read_ppm4(image_path)
    if w % 4 != 0 || h % 4 != 0
        error("image dimensions must be a multiple of 4")
    end
    @printf("Image Loaded '%s', %d x %d pixels\n\n", image_path, w, h)
    block_image = block_linear_image(data, w, h)
    blocks = UInt32((w ÷ 4) * (h ÷ 4))
    blocks_per_launch = min(Int(blocks), 768 * 24)
    @printf("Running DXT Compression on %u x %u image...\n", w, h)
    @printf("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", blocks, NUM_THREADS, blocks * NUM_THREADS)

    d_image = CuArray(block_image)
    d_result = CuArray(zeros(UInt32, Int(blocks) * 2))
    d_permutations = CuArray(compute_permutations())
    CUDA.synchronize()
    t0 = time_ns()
    threads = 128
    for _ in 1:num_iterations
        j = 0
        while j < Int(blocks)
            launch_blocks = min(blocks_per_launch, Int(blocks) - j)
            grid = cld(launch_blocks, threads)
            @cuda threads=threads blocks=grid compress_kernel!(d_permutations, d_image, d_result, Int32(launch_blocks), Int32(j))
            j += launch_blocks
        end
    end
    CUDA.synchronize()
    dt_us = (time_ns() - t0) * 1e-3 / num_iterations
    @printf("Average kernel execution time %f (us)\n", dt_us)

    h_result = Array(d_result)
    write_dds(output_name(image_path), h_result, w, h)
    reference = read_dds_payload(reference_path, length(h_result))
    @printf("\nChecking accuracy...\n")
    rms = 0.0f0
    for y in 0:4:(Int(h)-1), x in 0:4:(Int(w)-1)
        block_idx = (y ÷ 4) * Int(w ÷ 4) + (x ÷ 4) + 1
        rms += Float32(compare_block(h_result, reference, block_idx))
    end
    rms /= Float32(w * h * 3)
    @printf("RMS(reference, result) = %f\n\n", rms)
    println(rms <= ERROR_THRESHOLD ? "PASS" : "FAIL")
    return rms <= ERROR_THRESHOLD
end

function main()
    length(ARGS) >= 3 || error("usage: main.jl <image.ppm> <reference.dds> <iterations>")
    ok = run_case(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
    ok || exit(1)
end

main()
