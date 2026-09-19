using CUDA
using Printf

const CUDACore = CUDA.CUDACore

@inline function atomic_agg_inc!(d, idx::Int32)
    mask = UInt32(0)
    full_mask = UInt32(0xffffffff)
    lane0 = (threadIdx().x - Int32(1)) % Int32(32)

    for i in Int32(0):Int32(31)
        peer_idx = CUDACore.shfl_sync(full_mask, idx, i + Int32(1))
        peer_mask = CUDACore.vote_ballot_sync(full_mask, peer_idx == idx)
        if i == lane0
            mask = peer_mask
        end
    end

    leader0 = CUDACore.ffs(mask) - Int32(1)
    previous = Int32(0)

    if lane0 == leader0
        previous = CUDA.atomic_add!(pointer(d, idx), CUDACore.popc(mask))
    end

    previous = CUDACore.shfl_sync(mask, previous, leader0 + Int32(1))
    prior_lanes = mask & ((UInt32(1) << UInt32(lane0)) - UInt32(1))
    return previous + CUDACore.popc(prior_lanes)
end

function aggregate_kernel!(d, s::Int32)
    lane0 = threadIdx().x - Int32(1)
    idx = (lane0 % s) + Int32(1)  # CUDA's 0-based threadIdx.x % s, converted once to Julia indexing.
    atomic_agg_inc!(d, idx)
    return
end

function expected_count(block_size::Int, nblocks::Int, repeat::Int, ds::Int)
    wrapped = UInt32(block_size ÷ ds) * UInt32(nblocks) * UInt32(repeat)
    return reinterpret(Int32, wrapped)
end

function run(repeat::Int)
    nblocks = 65_536
    block_size = 256

    for ds in (32, 16, 8, 4, 2, 1)
        d = CUDA.zeros(Int32, ds)

        CUDA.synchronize()
        start_ns = time_ns()

        for _ in 1:repeat
            @cuda threads=block_size blocks=nblocks aggregate_kernel!(d, Int32(ds))
        end

        CUDA.synchronize()
        elapsed_s = (time_ns() - start_ns) * 1.0e-9
        @printf("Total kernel time (%d locations): %f (s)\n", ds, elapsed_s)

        h = Array(d)
        expected = expected_count(block_size, nblocks, repeat, ds)
        ok = all(x -> x == expected, h)
        println(ok ? "PASS" : "FAIL")
    end
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end

    repeat = parse(Int, ARGS[1])
    run(repeat)
    return 0
end

exit(main())
