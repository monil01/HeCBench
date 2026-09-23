using CUDA
using Printf
using Random

const MANUAL_VECTOR = 8
const NUM_THREADS_PER_WG = 64
const BLOOM_1 = 5
const BLOOM_2 = UInt32(0x7ffff)
const BLOOM_SIZE = 14
const DOC_ENDING_TAG = UInt32(0xffffffff)
const BLOCK_SIZE = 64
const DEFAULT_REPEAT = 100
const DEFAULT_NUM_DOCS = 256 * 1024
const DOC_LEN_SIGMA = 100.0f0
const AVG_DOC_LEN = 350.0f0

struct DataSet
    dimm1::Vector{UInt32}
    dimm2::Vector{UInt32}
    profile_weights::Vector{UInt64}
    doc_info::Vector{UInt64}
    profile_hash::Vector{UInt32}
    profile_score::Vector{UInt64}
    doc_sizes::Vector{UInt32}
    total_doc_size::UInt32
    total_doc_size_no_padding::UInt32
end

function parse_args(args)
    repeat = DEFAULT_REPEAT
    total_docs = DEFAULT_NUM_DOCS
    for arg in args
        if startswith(arg, "-p=")
            repeat = parse(Int, arg[4:end])
        elseif startswith(arg, "-n=")
            total_docs = parse(Int, arg[4:end])
        end
    end
    return repeat, total_docs
end

function rand_desh!(state::Vector{UInt32})
    z = state[1]
    w = state[2]
    z = UInt32(36969) * (z & UInt32(65535)) + (z >> 16)
    w = UInt32(18000) * (w & UInt32(65535)) + (w >> 16)
    state[1] = z
    state[2] = w
    return (z << 16) + w
end

function get_doc_length()
    len = round(Int, randn(Float32) * DOC_LEN_SIGMA + AVG_DOC_LEN)
    return UInt32(max(len, 10))
end

function setup_data(total_num_docs::Int)
    Random.seed!(2)
    rand_state = UInt32[1, 1]
    doc_sizes = Vector{UInt32}(undef, total_num_docs)
    starting_doc_id = Vector{UInt32}(undef, total_num_docs)
    doc_info = Vector{UInt64}(undef, total_num_docs)
    profile_score = fill(UInt64(typemax(UInt64)), total_num_docs)

    total_doc_size = UInt32(0)
    total_doc_size_no_padding = UInt32(0)
    for i in 1:total_num_docs
        unpadded = get_doc_length()
        size = unpadded & ~UInt32(2 * BLOCK_SIZE - 1)
        if (unpadded & UInt32(2 * BLOCK_SIZE - 1)) != 0
            size += UInt32(2 * BLOCK_SIZE)
        end
        starting_doc_id[i] = total_doc_size ÷ UInt32(2)
        start_line = UInt64(total_doc_size ÷ UInt32(2 * BLOCK_SIZE))
        end_line = start_line + UInt64(size ÷ UInt32(2 * BLOCK_SIZE)) - UInt64(1)
        doc_info[i] = (start_line << 32) | end_line
        total_doc_size += size
        total_doc_size_no_padding += unpadded
        doc_sizes[i] = unpadded
    end

    half_terms = Int(total_doc_size ÷ UInt32(2))
    @printf("Creating Documents total_terms=%d (no_pad=%d)\n", total_doc_size, total_doc_size_no_padding)
    dimm1 = fill(DOC_ENDING_TAG, half_terms)
    dimm2 = fill(DOC_ENDING_TAG, half_terms)

    for doc in 1:total_num_docs
        start = Int(starting_doc_id[doc])
        size = Int(doc_sizes[doc])
        for i0 in 0:(size ÷ 2 - 1)
            term = rand_desh!(rand_state) % UInt32((1 << 24) - 1)
            freq = (rand_desh!(rand_state) % UInt32(254)) + UInt32(1)
            dimm1[start + i0 + 1] = (term << 8) | freq

            term = rand_desh!(rand_state) % UInt32((1 << 24) - 1)
            freq = (rand_desh!(rand_state) % UInt32(254)) + UInt32(1)
            dimm2[start + i0 + 1] = (term << 8) | freq
        end
        if isodd(size)
            term = rand_desh!(rand_state) % UInt32((1 << 24) - 1)
            freq = (rand_desh!(rand_state) % UInt32(254)) + UInt32(1)
            dimm1[start + size ÷ 2 + 1] = (term << 8) | freq
        end
    end

    println("Creating Profile")
    profile_weights = zeros(UInt64, 1 << 24)
    profile_hash = zeros(UInt32, 1 << BLOOM_SIZE)
    for _ in 1:16384
        entry = rand_desh!(rand_state) % UInt32(1 << 24)
        profile_weights[Int(entry) + 1] = UInt64(10)
        hash1 = entry >> BLOOM_1
        profile_hash[Int(hash1 >> 5) + 1] |= UInt32(1) << (hash1 & UInt32(0x1f))
        hash2 = entry & BLOOM_2
        profile_hash[Int(hash2 >> 5) + 1] |= UInt32(1) << (hash2 & UInt32(0x1f))
    end

    return DataSet(dimm1, dimm2, profile_weights, doc_info, profile_hash,
                   profile_score, doc_sizes, total_doc_size, total_doc_size_no_padding)
end

@inline function mulfp(weight::UInt64, freq::UInt32)
    part1 = UInt32(weight & UInt64(0xfffff))
    part2 = UInt32((weight >> 24) & UInt64(0xffff))
    res1 = part1 * freq
    res2 = part2 * freq
    return UInt64(res1) + (UInt64(res2) << 24)
end

function compute_kernel!(d1, d2, weights1, weights2, bloom, partial_hi, partial_lo)
    partial = CUDA.@cuStaticSharedMem(UInt64, NUM_THREADS_PER_WG ÷ MANUAL_VECTOR)
    gid0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    sum = UInt64(0)
    base = gid0 * Int32(MANUAL_VECTOR)

    @inbounds for j in Int32(0):(Int32(MANUAL_VECTOR) - Int32(1))
        curr = d1[base + j + Int32(1)]
        if curr != DOC_ENDING_TAG
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((bloom[Int32(hash1 >> 5) + Int32(1)] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((bloom[Int32(hash2 >> 5) + Int32(1)] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                sum += mulfp(weights1[Int32(word_id) + Int32(1)], freq)
            end
        end
    end

    @inbounds for j in Int32(0):(Int32(MANUAL_VECTOR) - Int32(1))
        curr = d2[base + j + Int32(1)]
        if curr != DOC_ENDING_TAG
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((bloom[Int32(hash1 >> 5) + Int32(1)] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((bloom[Int32(hash2 >> 5) + Int32(1)] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                sum += mulfp(weights2[Int32(word_id) + Int32(1)], freq)
            end
        end
    end

    @inbounds partial[threadIdx().x] = sum
    sync_threads()
    if threadIdx().x == Int32(1)
        final = UInt64(0)
        @inbounds for i in Int32(1):Int32(NUM_THREADS_PER_WG ÷ MANUAL_VECTOR)
            final += partial[i]
        end
        @inbounds partial_hi[blockIdx().x] = UInt32(final >> 32)
        @inbounds partial_lo[blockIdx().x] = UInt32(final & UInt64(0xffffffff))
    end
    return
end

function reduction_kernel!(doc_info, partial_hi, partial_lo, result)
    gid0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    info = @inbounds doc_info[gid0 + Int32(1)]
    start = UInt32(info >> 32)
    stop = UInt32(info & UInt64(0xffffffff))
    total = UInt64(0)
    i = start
    @inbounds while i <= stop
        upper = UInt64(partial_hi[Int32(i) + Int32(1)])
        lower = UInt64(partial_lo[Int32(i) + Int32(1)])
        total += (upper << 32) | lower
        i += UInt32(1)
    end
    @inbounds result[gid0 + Int32(1)] = total
    return
end

function run_cpu(data::DataSet)
    cpu_score = zeros(UInt64, length(data.doc_sizes))
    total = UInt32(0)
    falsies = UInt32(0)
    for doc in eachindex(data.doc_sizes)
        info = data.doc_info[doc]
        start_line = Int(info >> 32)
        start = start_line * BLOCK_SIZE
        size = Int(data.doc_sizes[doc])
        score = UInt64(0)
        for i0 in 0:(size ÷ 2 + (size % 2) - 1)
            curr = data.dimm1[start + i0 + 1]
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((data.profile_hash[Int(hash1 >> 5) + 1] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((data.profile_hash[Int(hash2 >> 5) + 1] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                total += UInt32(1)
                data.profile_weights[Int(word_id) + 1] == 0 && (falsies += UInt32(1))
                score += data.profile_weights[Int(word_id) + 1] * UInt64(freq)
            end
        end
        for i0 in 0:(size ÷ 2 - 1)
            curr = data.dimm2[start + i0 + 1]
            freq = curr & UInt32(0xff)
            word_id = curr >> 8
            hash1 = word_id >> BLOOM_1
            hash2 = word_id & BLOOM_2
            inh1 = ((data.profile_hash[Int(hash1 >> 5) + 1] >> (hash1 & UInt32(0x1f))) & UInt32(1)) != 0
            inh2 = ((data.profile_hash[Int(hash2 >> 5) + 1] >> (hash2 & UInt32(0x1f))) & UInt32(1)) != 0
            if inh1 && inh2
                total += UInt32(1)
                data.profile_weights[Int(word_id) + 1] == 0 && (falsies += UInt32(1))
                score += data.profile_weights[Int(word_id) + 1] * UInt64(freq)
            end
        end
        cpu_score[doc] = score
    end
    @printf("total_access = %d , falsies = %d, percentage = %f hit= %g\n",
            total, falsies, Float32(total) / Float32(data.total_doc_size),
            Float32(total - falsies) / Float32(data.total_doc_size))
    for i in eachindex(cpu_score)
        if cpu_score[i] != data.profile_score[i]
            @printf("FAILED\n   : doc[%d] score: CPU = %lu, Device = %lu\n",
                    i - 1, cpu_score[i], data.profile_score[i])
            return false
        end
    end
    println("Verification: PASS")
    return true
end

function main()
    repeat, total_docs = parse_args(ARGS)
    @printf("Total number of documents: %u\n", total_docs)
    @printf("Kernel execution count: %u\n", repeat)
    @printf("RAND_MAX: %d\n", typemax(Int32))
    println("Allocating and setting up data")
    data = setup_data(total_docs)

    local_size = BLOCK_SIZE ÷ MANUAL_VECTOR
    global_size = Int(data.total_doc_size ÷ UInt32(2) ÷ UInt32(MANUAL_VECTOR) ÷ UInt32(local_size))
    global_size_reduction = cld(total_docs, BLOCK_SIZE)
    partial_len = Int(data.total_doc_size ÷ UInt32(2 * BLOCK_SIZE))

    d_d1 = CuArray(data.dimm1)
    d_d2 = CuArray(data.dimm2)
    d_w1 = CuArray(data.profile_weights)
    d_w2 = CuArray(data.profile_weights)
    d_hash = CuArray(data.profile_hash)
    d_docinfo = CuArray(data.doc_info)
    d_score = CUDA.zeros(UInt64, total_docs)
    d_hi = CUDA.zeros(UInt32, partial_len)
    d_lo = CUDA.zeros(UInt32, partial_len)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=local_size blocks=global_size compute_kernel!(d_d1, d_d2, d_w1, d_w2, d_hash, d_hi, d_lo)
        @cuda threads=BLOCK_SIZE blocks=global_size_reduction reduction_kernel!(d_docinfo, d_hi, d_lo, d_score)
    end
    CUDA.synchronize()
    kernel_s = (time_ns() - t0) / 1.0e9 / repeat
    println("======================================================")
    @printf("Kernel Time = %f ms (averaged over %d times)\n", kernel_s * 1000.0, repeat)
    @printf("Throughput = %f\n", Float64(data.total_doc_size_no_padding) / kernel_s / 1.0e6)

    copyto!(data.profile_score, d_score)
    println("Done")
    ok = run_cpu(data)
    return ok ? 0 : 1
end

exit(main())
