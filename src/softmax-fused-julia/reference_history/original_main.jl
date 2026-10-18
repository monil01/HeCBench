using CUDA
using Printf
using Random

const THREADS = 256

function scaled_masked_softmax_kernel!(dst, src, mask, scale::Float32,
                                       rows::Int64, key_seq_len::Int32,
                                       query_seq_len::Int32, pad_batches::Int32)
    row = blockIdx().x
    tid = threadIdx().x
    stride = blockDim().x
    if row <= rows
        q = Int32((row - Int32(1)) % query_seq_len + Int32(1))
        mask_row = pad_batches == Int32(1) ? q : row
        maxval = -10000.0f0
        for col in tid:stride:key_seq_len
            midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
            idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
            m = @inbounds mask[midx]
            v = m != UInt8(1) ? (@inbounds(src[idx]) * scale) : -10000.0f0
            maxval = max(maxval, v)
        end
        sh = CUDA.@cuStaticSharedMem(Float32, THREADS)
        sh[tid] = maxval
        sync_threads()
        offset = blockDim().x ÷ Int32(2)
        while offset > 0
            if tid <= offset
                sh[tid] = max(sh[tid], sh[tid + offset])
            end
            sync_threads()
            offset ÷= Int32(2)
        end
        maxval = sh[1]
        full_mask = maxval == -10000.0f0

        sumval = 0.0f0
        for col in tid:stride:key_seq_len
            midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
            idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
            m = @inbounds mask[midx]
            v = m != UInt8(1) ? (@inbounds(src[idx]) * scale) : -10000.0f0
            e = exp(v - maxval)
            sumval += e
        end
        sh[tid] = sumval
        sync_threads()
        offset = blockDim().x ÷ Int32(2)
        while offset > 0
            if tid <= offset
                sh[tid] += sh[tid + offset]
            end
            sync_threads()
            offset ÷= Int32(2)
        end
        sumval = sh[1]

        for col in tid:stride:key_seq_len
            out = 0.0f0
            if full_mask == false
                midx = (Int64(mask_row) - 1) * Int64(key_seq_len) + Int64(col)
                idx = (Int64(row) - 1) * Int64(key_seq_len) + Int64(col)
                m = @inbounds mask[midx]
                v = m != UInt8(1) ? (@inbounds(src[idx]) * scale) : -10000.0f0
                out = exp(v - maxval) / sumval
            end
            @inbounds dst[(Int64(row) - 1) * Int64(key_seq_len) + Int64(col)] = Float32(out)
        end
    end
    return
end

function reference!(out, inp, mask, scale, pad_batches, batches, heads, query_seq_len, key_seq_len)
    tmp = Vector{Float32}(undef, key_seq_len)
    rows = batches * heads * query_seq_len
    for row in 1:rows
        q = (row - 1) % query_seq_len + 1
        mask_row = pad_batches == 1 ? q : row
        base = (row - 1) * key_seq_len
        mbase = (mask_row - 1) * key_seq_len
        maxval = -10000.0f0
        for j in 1:key_seq_len
            tmp[j] = mask[mbase + j] != UInt8(1) ? inp[base + j] * scale : -10000.0f0
            maxval = max(maxval, tmp[j])
        end
        full_mask = maxval == -10000.0f0
        sumval = 0.0f0
        for j in 1:key_seq_len
            tmp[j] = exp(tmp[j] - maxval)
            sumval += tmp[j]
        end
        for j in 1:key_seq_len
            out[base + j] = full_mask ? 0.0f0 : tmp[j] / sumval
        end
    end
    return out
end

function make_case(batches, heads, query_seq_len, key_seq_len)
    rows = batches * heads * query_seq_len
    elems = rows * key_seq_len
    rng = MersenneTwister(123)
    mask = Vector{UInt8}(undef, elems)
    for row in 1:rows
        len = rand(rng, 0:(key_seq_len ÷ 2 - 1))
        base = (row - 1) * key_seq_len
        fill!(@view(mask[base + 1:base + len]), UInt8(1))
        fill!(@view(mask[base + len + 1:base + key_seq_len]), UInt8(0))
    end
    input = rand(rng, Float32, elems) .* 2.0f0 .- 1.0f0
    outliers = rand(rng, 0:(key_seq_len - 1), rows)
    for k in 1:query_seq_len, j in 1:(batches * heads)
        idx = (j - 1) * key_seq_len + outliers[(j - 1) * query_seq_len + k] + 1
        input[idx] *= 20.0f0
    end
    return input, mask
end

function run_one(batches, heads, query_seq_len, key_seq_len, pad_batches, repeat)
    rows = batches * heads * query_seq_len
    elems = rows * key_seq_len
    scale = inv(sqrt(Float32(key_seq_len)))
    input, mask = make_case(batches, heads, query_seq_len, key_seq_len)
    output_ref = Vector{Float32}(undef, elems)
    reference!(output_ref, input, mask, scale, pad_batches, batches, heads, query_seq_len, key_seq_len)

    d_input = CuArray(input)
    d_mask = CuArray(mask)
    d_output = CUDA.zeros(Float32, elems)
    @cuda threads=THREADS blocks=rows scaled_masked_softmax_kernel!(d_output, d_input, d_mask, scale,
                                                                    Int64(rows), Int32(key_seq_len),
                                                                    Int32(query_seq_len), Int32(pad_batches))
    output = Array(d_output)
    ok = all(abs.(output .- output_ref) .<= 1.0f-3)
    if !ok
        bad = findfirst(x -> abs(x[1] - x[2]) > 1.0f-3, zip(output, output_ref))
        @printf("Mismatch at index %d: %f %f\n", bad, output[bad], output_ref[bad])
    end
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=rows scaled_masked_softmax_kernel!(d_output, d_input, d_mask, scale,
                                                                        Int64(rows), Int32(key_seq_len),
                                                                        Int32(query_seq_len), Int32(pad_batches))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (ms)\n", (time_ns() - start) * 1.0e-6 / repeat)
    return ok
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <batch> <head> <query length> <key length> <repeat>")
        return 1
    end
    batches = parse(Int, args[1])
    heads = parse(Int, args[2])
    query_seq_len = parse(Int, args[3])
    key_seq_len = parse(Int, args[4])
    repeat = parse(Int, args[5])

    ok = true
    ok &= run_one(batches, heads, query_seq_len, key_seq_len, 0, repeat)
    ok &= run_one(batches, heads, query_seq_len, key_seq_len, 1, repeat)
    ok &= run_one(batches, heads, query_seq_len, key_seq_len, 0, repeat)
    ok &= run_one(batches, heads, query_seq_len, key_seq_len, 1, repeat)
    return ok ? 0 : 2
end

exit(main(ARGS))
