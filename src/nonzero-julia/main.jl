using CUDA
using Printf
using Random

function write_indices_kernel!(flat_indices, output, nrows::Int64, ncols::Int64, nzero::Int64)
    idx0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if idx0 < nzero
        div = Int64(1)
        idx_flat = @inbounds flat_indices[Int(idx0) + 1] - Int64(1)

        # CUDA's reverse dimension loop writes column first, then row.
        @inbounds output[Int(idx0 + nzero) + 1] = (idx_flat ÷ div) % ncols
        div *= ncols
        @inbounds output[Int(idx0) + 1] = (idx_flat ÷ div) % nrows
    end
    return
end

function fill_random!(input::Vector{T}, rng) where {T}
    nzeros = Int64(0)
    for i in eachindex(input)
        v = rand(rng, -1:1)
        input[i] = T(v)
        nzeros += v == 0 ? Int64(0) : Int64(1)
    end
    return nzeros
end

function run_nonzero(::Type{T}, nrows::Int, ncols::Int, repeat::Int) where {T}
    in_sizes = (Int64(nrows), Int64(ncols))
    num_items = in_sizes[1] * in_sizes[2]
    h_in = Vector{T}(undef, Int(num_items))
    rng = MersenneTwister(19937)

    ok = true
    sum_time_ns = Int128(0)
    idx_time_ns = Int128(0)

    for _ in 1:repeat
        r_nzeros = Int64(0)
        while r_nzeros == 0
            r_nzeros = fill_random!(h_in, rng)
        end

        d_in = CuArray(h_in)

        CUDA.synchronize()
        t0 = time_ns()
        h_nzeros = Int64(CUDA.count(!iszero, d_in))
        CUDA.synchronize()
        sum_time_ns += Int128(time_ns() - t0)

        if h_nzeros != r_nzeros
            @printf("Number of non-zero elements mismatch: %ld != %ld (expected)\n",
                    h_nzeros, r_nzeros)
            ok = false
        else
            d_out = CUDA.zeros(Int64, Int(h_nzeros * Int64(2)))
            threads = 256

            CUDA.synchronize()
            t1 = time_ns()
            d_flat = CUDA.findall(!iszero, d_in)

            index_blocks = cld(Int(h_nzeros), threads)
            @cuda threads=threads blocks=index_blocks write_indices_kernel!(
                d_flat, d_out, in_sizes[1], in_sizes[2], h_nzeros)
            CUDA.synchronize()
            idx_time_ns += Int128(time_ns() - t1)

            h_out = Array(d_out)
            cnt_nzero = Int64(0)
            for i in 1:Int(h_nzeros)
                row = h_out[i]
                col = h_out[Int(h_nzeros) + i]
                src_idx = in_sizes[2] * row + col + Int64(1)
                if h_in[Int(src_idx)] != zero(T)
                    cnt_nzero += Int64(1)
                end
            end
            ok = cnt_nzero == h_nzeros
        end

        ok || break
    end

    @printf("Average time for sum reduction: %lf (us)\n",
            Float64(sum_time_ns) * 1e-3 / repeat)
    @printf("Average time for write index operations: %lf (us)\n",
            Float64(idx_time_ns) * 1e-3 / repeat)
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of rows> <number of columns> <repeat>")
        return 1
    end

    nrows = parse(Int, args[1])
    ncols = parse(Int, args[2])
    repeat = parse(Int, args[3])
    nrows = nrows <= 0 ? 1 : nrows
    ncols = ncols <= 0 ? 1 : ncols

    for _ in 1:2
        println("=========== Data type is I8 ==========")
        run_nonzero(Int8, nrows, ncols, repeat)

        println("=========== Data type is I16 ==========")
        run_nonzero(Int16, nrows, ncols, repeat)

        println("=========== Data type is I32 ==========")
        run_nonzero(Int32, nrows, ncols, repeat)

        println("=========== Data type is FP32 ==========")
        run_nonzero(Float32, nrows, ncols, repeat)

        println("=========== Data type is FP64 ==========")
        run_nonzero(Float64, nrows, ncols, repeat)
    end

    return 0
end

exit(main(ARGS))
