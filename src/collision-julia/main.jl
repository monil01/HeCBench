using CUDA
using Printf
using Random

const WARP_SIZE = 32

function check_duplicates_kernel(values, flags)
    dup = false
    @inbounds for i in 1:WARP_SIZE
        vi = values[i]
        for j in (i + 1):WARP_SIZE
            dup |= vi == values[j]
        end
    end
    flags[threadIdx().x] = dup ? Int32(1) : Int32(0)
    return
end

function check_mask_kernel(values, mask_out)
    if threadIdx().x == 1
        mask = UInt32(0)
        @inbounds for lane in 1:WARP_SIZE
            val = values[lane]
            seen = false
            for other in 1:(lane - 1)
                seen |= values[other] == val
            end
            if seen
                mask |= UInt32(1) << UInt32(lane - 1)
            end
        end
        mask_out[1] = mask
    end
    return
end

function unique_prefix(n)
    vals = Int32[]
    seen = Set{Int32}()
    while length(vals) < n
        r = Int32(rand(0:typemax(Int32)))
        if !(r in seen)
            push!(vals, r)
            push!(seen, r)
        end
    end
    return vals
end

function make_values(num_dups)
    vals = unique_prefix(WARP_SIZE - num_dups)
    append!(vals, fill(vals[1], num_dups))
    return vals
end

function check_duplicates(values)
    d_values = CuArray(values)
    d_flags = CUDA.zeros(Int32, WARP_SIZE)
    @cuda threads=WARP_SIZE check_duplicates_kernel(d_values, d_flags)
    synchronize()
    return Array(d_flags)
end

function check_duplicate_mask(values)
    d_values = CuArray(values)
    d_mask = CUDA.zeros(UInt32, 1)
    @cuda threads=WARP_SIZE check_mask_kernel(d_values, d_mask)
    synchronize()
    return Array(d_mask)[1]
end

function test_collision()
    for num_dups in 0:(WARP_SIZE - 1)
        vals = make_values(num_dups)
        flags = check_duplicates(vals)
        expected = num_dups > 0 ? Int32(1) : Int32(0)
        all(flags .== expected) || error("duplicate flag mismatch for numDups=$num_dups")
    end
end

function test_collision_mask()
    for num_dups in 0:(WARP_SIZE - 1)
        vals = make_values(num_dups)
        mask = check_duplicate_mask(vals)
        expected = num_dups > 0 ? (typemax(UInt32) << UInt32(WARP_SIZE - num_dups)) : UInt32(0)
        mask == expected || error("numDups=$num_dups expected=$(string(expected, base=16)) mask=$(string(mask, base=16))")
    end
end

function main()
    length(ARGS) == 1 || begin
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    repeat > 0 || error("repeat must be positive")
    CUDA.allowscalar(false)
    Random.seed!(123)

    start = time_ns()
    for _ in 1:repeat
        test_collision()
    end
    elapsed = time_ns() - start
    @printf("Average execution time of the function test_collision: %f (us)\n", elapsed * 1.0e-3 / repeat)

    start = time_ns()
    for _ in 1:repeat
        test_collision_mask()
    end
    elapsed = time_ns() - start
    @printf("Average execution time of the function test_collisionMask: %f (us)\n", elapsed * 1.0e-3 / repeat)
    println("PASS")
end

main()
