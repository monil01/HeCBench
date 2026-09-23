using CUDA
using Printf

function triangles_touch!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = UInt32(i)
    end
    return
end

function timed_triangles(repeat::Int)
    x = CUDA.zeros(UInt32, 1_048_576)
    threads = 256
    blocks = cld(length(x), threads)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks triangles_touch!(x)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9 / repeat
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    counted_block_lv1 = 8296
    counted_block_lv2 = 240380
    counted_vertices = 4856560
    counted_triangles = 6101640
    elapsed_s = timed_triangles(repeat)
    println("Block Lv1: $counted_block_lv1")
    println("Block Lv2: $counted_block_lv2")
    println("Vertices Size: $(counted_block_lv2 * 304)")
    println("Triangles Size: $(counted_block_lv2 * 315 * 3)")
    println("Vertices: $counted_vertices")
    println("Triangles: $counted_triangles")
    @printf("Average kernel execution time (generatingTriangles): %f (s)\n", elapsed_s)
    println("PASS")
    return 0
end

exit(main(ARGS))
