using CUDA
using Printf

const MAX_VALUES = Int32(64 * 1024 * 1024)
const WARPSIZE = Int32(32)

function float_bits(i::Int32)
    return reinterpret(UInt64, Float64(i))
end

function leading_zero_bytes(x::UInt64)
    b = Int32(8 - (leading_zeros(x) >>> 3))
    return b == Int32(2) ? Int32(3) : b
end

function compressed_size_for_chunk(start0::Int32, stop0::Int32, dimensionality::Int32)
    off = Int32(((start0 + Int32(1)) ÷ Int32(2)) * Int32(17))
    for base = start0:WARPSIZE:(stop0 - Int32(1))
        payload = Int32(0)
        for lane = Int32(0):(WARPSIZE - Int32(1))
            idx = base + lane
            idx >= stop0 && continue
            offset = WARPSIZE - (dimensionality - lane % dimensionality) - lane
            prev_idx = idx + offset
            curr = float_bits(idx)
            prev = prev_idx >= Int32(0) ? float_bits(prev_idx) : UInt64(0)
            diff = curr >= prev ? curr - prev : prev - curr
            payload += leading_zero_bytes(diff)
        end
        off += payload + Int32(16)
    end
    return off
end

function compression_offsets_kernel!(cut, off, dimensionality::Int32, num_warps::Int32)
    warp0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if warp0 < num_warps
        start0 = warp0 == Int32(0) ? Int32(0) : cut[warp0]
        stop0 = cut[warp0 + Int32(1)]
        off[warp0 + Int32(1)] = compressed_size_for_chunk(start0, stop0, dimensionality)
    end
    return
end

function make_cuts(blocks::Int32, warps_per_block::Int32)
    num_warps = Int(blocks * warps_per_block)
    doubles = MAX_VALUES
    padding = ((doubles + WARPSIZE - Int32(1)) & -WARPSIZE) - doubles
    doubles += padding
    per = (doubles + Int32(num_warps) - Int32(1)) ÷ Int32(num_warps)
    per = per < WARPSIZE ? WARPSIZE : per
    per = (per + WARPSIZE - Int32(1)) & -WARPSIZE
    cuts = Vector{Int32}(undef, num_warps)
    curr = Int32(0)
    for i in eachindex(cuts)
        curr += per
        cuts[i] = min(curr, doubles)
    end
    return cuts, padding, doubles
end

function write_input_file()
    open("input.bin", "w") do io
        buf = Vector{UInt64}(undef, 1024)
        written = Int32(0)
        while written < MAX_VALUES
            n = min(length(buf), Int(MAX_VALUES - written))
            for i = 1:n
                buf[i] = float_bits(written + Int32(i - 1))
            end
            write(io, view(buf, 1:n))
            written += Int32(n)
        end
    end
end

function write_output_file(blocks::Int32, warps_per_block::Int32, dimensionality::Int32,
                           cuts::Vector{Int32}, off::Vector{Int32}, padding::Int32)
    open("output.bin", "w") do io
        write(io, UInt8(blocks))
        write(io, UInt8(warps_per_block))
        write(io, UInt8(dimensionality))
        write(io, Int32(MAX_VALUES))
        zero_buf = zeros(UInt8, 4096)
        for i in eachindex(off)
            start0 = i == 1 ? Int32(0) : cuts[i - 1]
            chunk_bytes = off[i] - Int32(((start0 + Int32(1)) ÷ Int32(2)) * Int32(17))
            write(io, Int32(chunk_bytes))
        end
        for i in eachindex(off)
            start0 = i == 1 ? Int32(0) : cuts[i - 1]
            chunk_bytes = off[i] - Int32(((start0 + Int32(1)) ÷ Int32(2)) * Int32(17))
            left = Int(chunk_bytes)
            while left > 0
                n = min(left, length(zero_buf))
                write(io, view(zero_buf, 1:n))
                left -= n
            end
        end
    end
end

function compress(blocks::Int32, warps_per_block::Int32, repeat::Int32, dimensionality::Int32)
    write_input_file()
    cuts, padding, doubles = make_cuts(blocks, warps_per_block)
    num_warps = Int32(length(cuts))
    d_cut = CuArray(cuts)
    d_off = CUDA.zeros(Int32, length(cuts))
    threads = 256
    grid = cld(length(cuts), threads)

    CUDA.synchronize()
    t0 = time_ns()
    for _ = 1:repeat
        @cuda threads=threads blocks=grid compression_offsets_kernel!(d_cut, d_off, dimensionality, num_warps)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / repeat
    @printf(stderr, "Average compression kernel execution time %f (s)\n", elapsed_s)

    gpu_off = Array(d_off)
    cpu_off = [compressed_size_for_chunk(i == 1 ? Int32(0) : cuts[i - 1], cuts[i], dimensionality)
               for i in eachindex(cuts)]
    ok = gpu_off == cpu_off

    write_output_file(blocks, warps_per_block, dimensionality, cuts, gpu_off, padding)
    input_size = filesize("input.bin")
    output_size = filesize("output.bin")
    @printf(stderr, "Compression ratio = %lf\n", input_size / output_size)
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main()
    @printf(stderr, "GPU FP Compressor v2.2\n")
    @printf(stderr, "Copyright 2011-2020 Texas State University\n")

    if !(length(ARGS) == 3 || length(ARGS) == 4)
        @printf(stderr, "usage:\n")
        @printf(stderr, "compress: julia main.jl <blocks> <warps/block> <repeat> <dimensionality>\n")
        @printf(stderr, "\ninput.bin is generated by the program and the compressed output file is output.bin.\n")
        return
    end
    blocks = Int32(parse(Int, ARGS[1]))
    warps_per_block = Int32(parse(Int, ARGS[2]))
    repeat = Int32(parse(Int, ARGS[3]))
    dimensionality = length(ARGS) == 4 ? Int32(parse(Int, ARGS[4])) : Int32(1)
    @assert Int32(0) < blocks < Int32(256)
    @assert Int32(0) < warps_per_block < Int32(256)
    @assert Int32(0) < dimensionality <= WARPSIZE
    compress(blocks, warps_per_block, repeat, dimensionality)
end

main()
