using CUDA
using Printf

const D_FACTOR = 0.85f0
const MAX_ITER = 1000
const THRESHOLD = 1.0f-16
const BLOCK_SIZE = 256

function map_kernel!(pages, page_ranks, maps, noutlinks, n::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 < n
        outbound_rank = page_ranks[i0 + Int32(1)] / Float32(noutlinks[i0 + Int32(1)])
        base = i0 * n
        for j0 in Int32(0):(n - Int32(1))
            @inbounds maps[base + j0 + Int32(1)] = Float32(pages[base + j0 + Int32(1)]) * outbound_rank
        end
    end
    return
end

function reduce_kernel!(page_ranks, maps, n::Int32, diffs)
    j0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if j0 < n
        old_rank = page_ranks[j0 + Int32(1)]
        new_rank = 0.0f0
        for i0 in Int32(0):(n - Int32(1))
            @inbounds new_rank += maps[i0 * n + j0 + Int32(1)]
        end
        new_rank = ((1.0f0 - D_FACTOR) / Float32(n)) + D_FACTOR * new_rank
        @inbounds diffs[j0 + Int32(1)] = max(abs(new_rank - old_rank), diffs[j0 + Int32(1)])
        @inbounds page_ranks[j0 + Int32(1)] = new_rank
    end
    return
end

function libc_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function random_pages(n::Int, divisor::Int)
    if divisor <= 0
        error("Invalid divisor")
    end
    pages = zeros(Int32, n * n)
    noutlinks = zeros(UInt32, n)
    for i in 0:n-1
        for j in 0:n-1
            if i != j && mod(abs(libc_rand()), divisor) == 0
                pages[i * n + j + 1] = 1
                noutlinks[i + 1] += UInt32(1)
            end
        end
        if noutlinks[i + 1] == 0
            k = 0
            while true
                k = mod(abs(libc_rand()), n)
                k != i && break
            end
            pages[i * n + k + 1] = 1
            noutlinks[i + 1] = UInt32(1)
        end
    end
    return pages, noutlinks
end

maximum_dif(diffs) = maximum(diffs)

function map_ref!(pages, page_ranks, maps, noutlinks, n)
    for i in 0:n-1
        outbound = page_ranks[i + 1] / Float32(noutlinks[i + 1])
        for j in 0:n-1
            maps[i * n + j + 1] = Float32(pages[i * n + j + 1]) * outbound
        end
    end
end

function reduce_ref!(page_ranks, maps, n, diffs)
    for j in 0:n-1
        old_rank = page_ranks[j + 1]
        new_rank = 0.0f0
        for i in 0:n-1
            new_rank += maps[i * n + j + 1]
        end
        new_rank = ((1.0f0 - D_FACTOR) / Float32(n)) + D_FACTOR * new_rank
        diffs[j + 1] = max(abs(new_rank - old_rank), diffs[j + 1])
        page_ranks[j + 1] = new_rank
    end
end

function parse_args()
    n = 1000
    iter = MAX_ITER
    thresh = THRESHOLD
    divisor = 2
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "-n"
            i += 1; n = parse(Int, ARGS[i])
        elseif ARGS[i] == "-i"
            i += 1; iter = parse(Int, ARGS[i])
        elseif ARGS[i] == "-t"
            i += 1; thresh = parse(Float32, ARGS[i])
        elseif ARGS[i] == "-q"
            i += 1; divisor = parse(Int, ARGS[i])
        else
            println(stderr, "Usage: main.jl [-n number of pages] [-i max iterations] [-t threshold] [-q divisor for zero density]")
            return nothing
        end
        i += 1
    end
    return n, iter, thresh, divisor
end

function main()
    parsed = parse_args()
    parsed === nothing && return 1
    n, iter, thresh, divisor = parsed
    libc_srand(UInt32(1))
    pages, noutlinks = random_pages(n, divisor)
    page_ranks = fill(1.0f0 / Float32(n), n)
    page_ranks_ref = copy(page_ranks)
    maps = Vector{Float32}(undef, n * n)
    maps_ref = Vector{Float32}(undef, n * n)
    diffs = zeros(Float32, n)
    diffs_ref = zeros(Float32, n)

    d_pages = CuArray(pages)
    d_ranks = CuArray(page_ranks)
    d_maps = CUDA.zeros(Float32, n * n)
    d_links = CuArray(noutlinks)
    d_diffs = CUDA.zeros(Float32, n)
    block_size = min(n, BLOCK_SIZE)
    num_blocks = cld(n, block_size)
    max_diff = 99.0f0
    ktime = 0.0
    println("Threshold: $thresh")
    t = 1
    while t <= iter && max_diff >= thresh
        CUDA.synchronize()
        start = time()
        @cuda threads=block_size blocks=num_blocks map_kernel!(d_pages, d_ranks, d_maps, d_links, Int32(n))
        @cuda threads=block_size blocks=num_blocks reduce_kernel!(d_ranks, d_maps, Int32(n), d_diffs)
        CUDA.synchronize()
        ktime += time() - start
        diffs = Array(d_diffs)
        max_diff = maximum_dif(diffs)
        t += 1
    end
    println(stderr, "Max difference $max_diff is reached at iteration $t")
    @printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",
            n, iter, thresh, ktime)

    max_diff_ref = 99.0f0
    t_ref = 1
    while t_ref <= iter && max_diff_ref >= thresh
        map_ref!(pages, page_ranks_ref, maps_ref, noutlinks, n)
        reduce_ref!(page_ranks_ref, maps_ref, n, diffs_ref)
        max_diff_ref = maximum_dif(diffs_ref)
        t_ref += 1
    end
    ok = abs(max_diff - max_diff_ref) < 1.0f-3
    println(ok ? "PASS" : "FAIL")
    return 0
end

exit(main())
