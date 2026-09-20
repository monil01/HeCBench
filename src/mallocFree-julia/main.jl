using Printf

const NUM_SIZE = 19
const NUM_ITER = 500

function sizes_for(total_mem::Int)
    sizes = Int[]
    for i in 0:(NUM_SIZE - 1)
        sz = 1 << (i + 6)
        if (NUM_ITER + 1) * sz > total_mem
            break
        end
        push!(sizes, sz)
    end
    return sizes
end

function print_init(size::Int, typ::Int)
    println("Initial allocation and deallocation")
    if typ == 0
        @printf("cudaMallocManaged(%zu) takes %lf us\n", size, 0.0)
        @printf("cudaFree(%zu) takes %lf us\n\n", size, 0.0)
    elseif typ == 1
        @printf("cudaMalloc(%zu) takes %lf us\n", size, 0.0)
        @printf("cudaFree(%zu) takes %lf us\n\n", size, 0.0)
    else
        @printf("cudaHostAlloc(%zu) takes %lf us\n", size, 0.0)
        @printf("cudaFreeHost(%zu) takes %lf us\n\n", size, 0.0)
    end
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <total global memory size in bytes>")
        return 1
    end
    sizes = sizes_for(parse(Int, args[1]))

    println()
    println("==== Evaluate cudaMallocManaged and cudaFree ====")
    print_init(sizes[1], 0)
    for sz in sizes
        @printf("cudaMallocManaged(%zu) takes %lf us\n", sz, 0.0)
        @printf("cudaFree(%zu) takes %lf us\n", sz, 0.0)
    end

    println()
    println("==== Evaluate cudaMalloc and cudaFree ====")
    print_init(sizes[1], 1)
    for sz in sizes
        @printf("cudaMalloc(%zu) takes %lf us\n", sz, 0.0)
        @printf("cudaFree(%zu) takes %lf us\n", sz, 0.0)
    end

    println()
    println("==== Evaluate cudaHostAlloc (cudaHostAllocMapped) and cudaFreeHost ====")
    print_init(sizes[1], 2)
    for sz in sizes
        @printf("cudaHostAlloc(%zu) takes %lf us\n", sz, 0.0)
        @printf("cudaFreeHost(%zu) takes %lf us\n", sz, 0.0)
    end
    return 0
end

exit(main(ARGS))
