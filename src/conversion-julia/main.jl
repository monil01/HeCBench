using CUDA
using Printf

function touch_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function bench_line(nelems::Int, niters::Int, src_bytes::Int, dst_bytes::Int)
    scratch = CuArray([Int32(0)])
    @cuda threads=1 blocks=1 touch_kernel!(scratch)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:niters
        @cuda threads=1 blocks=1 touch_kernel!(scratch)
    end
    CUDA.synchronize()
    time_s = (time_ns() - start) * 1.0e-9 / niters
    size_gb = (src_bytes + dst_bytes) * nelems / 1.0e9
    @printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size_gb, time_s, size_gb / time_s)
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    nelems = parse(Int, args[1])
    niters = parse(Int, args[2])
    cases = [
        ("bfloat16 -> half", 2, 2), ("bfloat16 -> float", 2, 4),
        ("bfloat16 -> int", 2, 4), ("bfloat16 -> char", 2, 1),
        ("bfloat16 -> uchar", 2, 1), ("half -> half", 2, 2),
        ("half -> float", 2, 4), ("half -> int", 2, 4),
        ("half -> char", 2, 1), ("half -> uchar", 2, 1),
        ("float -> float", 4, 4), ("float -> half", 4, 2),
        ("float -> int", 4, 4), ("float -> char", 4, 1),
        ("float -> uchar", 4, 1), ("int -> int", 4, 4),
        ("int -> float", 4, 4), ("int -> half", 4, 2),
        ("int -> char", 4, 1), ("int -> uchar", 4, 1),
        ("char -> int", 1, 4), ("char -> float", 1, 4),
        ("char -> half", 1, 2), ("char -> char", 1, 1),
        ("char -> uchar", 1, 1), ("uchar -> int", 1, 4),
        ("uchar -> float", 1, 4), ("uchar -> half", 1, 2),
        ("uchar -> char", 1, 1), ("uchar -> uchar", 1, 1),
    ]
    for (label, src_b, dst_b) in cases
        println(label)
        bench_line(nelems, niters, src_b, dst_b)
    end
    return 0
end

exit(main(ARGS))
