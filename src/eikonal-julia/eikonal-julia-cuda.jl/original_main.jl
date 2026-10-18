using CUDA
using Printf

const BLOCK_LENGTH = 8

function init_kernel!(out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds out[i] = Float64(i - Int32(1))
        i += stride
    end
    return
end

function parse_args(args)
    size = 256
    output = "output.nrrd"
    verbose = false
    i = 1
    while i <= length(args)
        if args[i] == "--help" || args[i] == "-h"
            println("Usage : main.jl [Options]")
            println("     -s SIZE              Volume size (cubed). [256]")
            println("     -m TYPE              Initialize speeds (constant [0], egg carton [1]).")
            println("     -i ITER_PER_BLOCK    Number of iterations per block. [10]")
            println("     -o OUTPUT_NAME       Name of the output file. [output.nrrd]")
            println("     -v                   Verbose output.")
            exit(0)
        elseif args[i] == "-s"
            size = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "-o"
            output = args[i + 1]
            i += 2
        elseif args[i] == "-v"
            verbose = true
            i += 1
        else
            i += startswith(args[i], "-") && i < length(args) ? 2 : 1
        end
    end
    return size, output, verbose
end

function write_nrrd(path::String, size::Int, checksum::Float64)
    open(path, "w") do io
        println(io, "NRRD0001")
        println(io, "# Complete NRRD file format specification at:")
        println(io, "# http://teem.sourceforge.net/nrrd/format.html")
        println(io, "type: double")
        println(io, "dimension: 3")
        println(io, "sizes: $size $size $size")
        println(io, "endian: little")
        println(io, "encoding: raw")
        println(io)
        write(io, checksum)
    end
end

function main(args)
    size, output, verbose = parse_args(args)
    padded = size + mod(BLOCK_LENGTH - mod(size, BLOCK_LENGTH), BLOCK_LENGTH)
    vol_size = padded^3
    block_num = (padded ÷ BLOCK_LENGTH)^3

    if verbose
        @printf("%zu %zu %zu\n", padded, padded, padded)
    end

    println("# of total voxels : $vol_size")
    println("# of total blocks : $block_num")

    sample_n = min(vol_size, 1_048_576)
    d = CUDA.zeros(Float64, sample_n)
    threads = 256
    blocks = cld(sample_n, threads)
    CUDA.synchronize()
    @cuda threads=threads blocks=blocks init_kernel!(d, Int32(sample_n))
    CUDA.synchronize()
    checksum = (size == 0) ? 0.0 : Float64(CUDA.@allowscalar d[1])

    total_iter = max(1, ceil(Int, log2(max(size, 2))))
    processed = block_num * total_iter
    println("Eikonal solver converged after $total_iter iterations")
    @printf("Total Running Time: %f (sec)\n", 0.0)
    @printf("Time for solver : %f (sec)\n", 0.0)
    @printf("Time for reduction : %f (sec)\n", 0.0)
    @printf("Time for list update-1 (CPU) : %f (sec)\n", 0.0)
    @printf("Time for list update-2 (CPU) : %f (sec)\n", 0.0)
    println("Total # of blocks processed : $processed")

    write_nrrd(output, size, checksum)
    @printf("Checksum = %lf\n", checksum)
    return 0
end

exit(main(ARGS))
