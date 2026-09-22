using CUDA
using Printf

const THREADS_PER_SITE = 36

@inline function link_index(site0::Int32, dir0::Int32, row0::Int32, col0::Int32)
    return (((site0 * Int32(4) + dir0) * Int32(3) + row0) * Int32(3) + col0) + Int32(1)
end

@inline function b_index(dir0::Int32, row0::Int32, col0::Int32)
    return ((dir0 * Int32(3) + row0) * Int32(3) + col0) + Int32(1)
end

function su3_mat_nn_kernel!(a_re, a_im, b_re, b_im, c_re, c_im, total_sites::Int32)
    id0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    site0 = id0 ÷ Int32(36)
    if site0 < total_sites
        local0 = id0 % Int32(36)
        dir0 = local0 ÷ Int32(9)
        row0 = (local0 % Int32(9)) ÷ Int32(3)
        col0 = local0 % Int32(3)
        sum_re = 0.0f0
        sum_im = 0.0f0
        for m0 in Int32(0):Int32(2)
            aidx = link_index(site0, dir0, row0, m0)
            bidx = b_index(dir0, m0, col0)
            ar = @inbounds a_re[aidx]
            ai = @inbounds a_im[aidx]
            br = @inbounds b_re[bidx]
            bi = @inbounds b_im[bidx]
            sum_re += ar * br - ai * bi
            sum_im += ar * bi + ai * br
        end
        cidx = link_index(site0, dir0, row0, col0)
        @inbounds c_re[cidx] = sum_re
        @inbounds c_im[cidx] = sum_im
    end
    return
end

function parse_args(args)
    iterations = 100
    ldim = 32
    threads = 128
    verbose = 1
    warmups = 1
    i = 1
    while i <= length(args)
        opt = args[i]
        if opt == "-i"
            i += 1; iterations = parse(Int, args[i])
        elseif opt == "-l"
            i += 1; ldim = parse(Int, args[i])
        elseif opt == "-t"
            i += 1; threads = parse(Int, args[i])
        elseif opt == "-v"
            i += 1; verbose = parse(Int, args[i])
        elseif opt == "-w"
            i += 1; warmups = parse(Int, args[i])
        elseif opt == "-h"
            println(stderr, "Usage: main.jl [-i iterations] [-l lattice dimension] [-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]")
            return nothing
        elseif opt == "-d"
            i += 1
        end
        i += 1
    end
    return iterations, ldim, threads, verbose, warmups
end

function run_su3(total_sites::Int, iterations::Int, threads::Int, warmups::Int, verbose::Int)
    elems = total_sites * 36
    a_re = fill(1.0f0, elems)
    a_im = zeros(Float32, elems)
    b_re = fill(1.0f0 / 3.0f0, 36)
    b_im = zeros(Float32, 36)
    c_re = zeros(Float32, elems)
    c_im = zeros(Float32, elems)

    d_a_re = CuArray(a_re)
    d_a_im = CuArray(a_im)
    d_b_re = CuArray(b_re)
    d_b_im = CuArray(b_im)
    d_c_re = CuArray(c_re)
    d_c_im = CuArray(c_im)

    blocks = total_sites
    if verbose >= 1
        @printf("Number of blocks set to %d\n", blocks)
        @printf("Threads per block set to %d\n", threads == 0 ? THREADS_PER_SITE : threads)
    end

    use_threads = threads == 0 ? THREADS_PER_SITE : threads
    CUDA.synchronize()
    start = time_ns()
    for iter in 0:(iterations + warmups - 1)
        if iter == warmups
            CUDA.synchronize()
            start = time_ns()
        end
        @cuda threads=use_threads blocks=blocks su3_mat_nn_kernel!(
            d_a_re, d_a_im, d_b_re, d_b_im, d_c_re, d_c_im, Int32(total_sites))
    end
    CUDA.synchronize()
    total_s = (time_ns() - start) * 1.0e-9

    c_re .= Array(d_c_re)
    c_im .= Array(d_c_im)
    max_error = 0.0f0
    @inbounds for idx in eachindex(c_re)
        max_error = max(max_error, abs(c_re[idx] - 1.0f0), abs(c_im[idx]))
    end
    return total_s, max_error, length(a_re), length(b_re), length(c_re)
end

function main(args)
    parsed = parse_args(args)
    parsed === nothing && return 1
    iterations, ldim, threads, verbose, warmups = parsed
    total_sites = ldim^4

    if verbose >= 1
        @printf("Number of sites = %d^4\n", ldim)
        @printf("Executing %d iterations with %d warmups\n", iterations, warmups)
        if threads != 0
            @printf("Threads per group = %d\n", threads)
        end
    end

    total_s, max_error, a_len, b_len, c_len = run_su3(total_sites, iterations, threads, warmups, verbose)
    if verbose >= 1
        @printf("Total kernel execution time = %f (s)\n", total_s)
    end

    tflop = Float64(iterations) * Float64(total_sites) * 864.0
    @printf("Total GFLOP/s = %.3f\n", tflop / total_s / 1.0e9)
    memory_usage = Float64((a_len + c_len + b_len) * sizeof(Float32) * 2)
    @printf("Total GByte/s (GPU memory)  = %.3f\n", Float64(iterations) * memory_usage / total_s / 1.0e9)
    if max_error <= 1.0f-6
        println("PASS")
    else
        @printf("Max SU3 error = %.9g\n", max_error)
        println("FAIL")
        return 1
    end
    if verbose >= 2
        @printf("Total allocation for matrices = %.3f MiB\n", memory_usage / 1048576.0)
        @printf("Approximate memory usage = %.3f MiB\n", memory_usage / 1048576.0)
    end
    return 0
end

exit(main(ARGS))
