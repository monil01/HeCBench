using CUDA
using Printf

function flip_kernel!(input, output, n::Int64, dim_size::Int64)
    idx0 = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if idx0 >= n
        return
    end
    i0 = idx0 ÷ (dim_size * dim_size)
    rem = idx0 - i0 * dim_size * dim_size
    i1 = rem ÷ dim_size
    i2 = rem - i1 * dim_size
    src = (dim_size - 1 - i0) * dim_size * dim_size +
          (dim_size - 1 - i1) * dim_size +
          (dim_size - 1 - i2)
    @inbounds output[idx0 + 1] = input[src + 1]
    return
end

function property(name, values)
    print("$name: ( ")
    for v in values
        print("$v ")
    end
    println(")")
end

function run_flip(::Type{T}, num_dims::Int, dim_size::Int, repeat::Int) where {T}
    shape = fill(Int64(dim_size), num_dims)
    flip_dims = collect(Int64(0):Int64(num_dims - 1))
    stride = [Int64(dim_size * dim_size), Int64(dim_size), Int64(1)]
    property("shape", shape)
    property("flip_dims", flip_dims)
    property("stride", stride)

    n = dim_size^num_dims
    input = T.(0:(n - 1))
    output = Vector{T}(undef, n)
    ref = reverse(reshape(input, ntuple(_ -> dim_size, num_dims)); dims=Tuple(1:num_dims))
    ref_vec = collect(reshape(ref, :))

    d_input = CuArray(input)
    d_output = CUDA.zeros(T, n)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks flip_kernel!(d_input, d_output, Int64(n), Int64(dim_size))
    CUDA.synchronize()
    copyto!(output, d_output)
    println(output == ref_vec ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks flip_kernel!(d_input, d_output, Int64(n), Int64(dim_size))
    end
    CUDA.synchronize()
    @printf("Average execution time of the flip kernel: %f (ms)\n", (time_ns() - start) * 1.0e-6 / repeat)
    return output == ref_vec
end

function main(args)
    if length(args) != 3
        println("Usage: ./main <number of dimensions> <size of each dimension> <repeat>")
        return 1
    end
    num_dims = parse(Int, args[1])
    dim_size = parse(Int, args[2])
    repeat = parse(Int, args[3])
    if num_dims != 3
        println("This Julia port supports the benchmark's 3D configuration")
        return 1
    end
    println("=========== Data type is FP32 ==========")
    ok = run_flip(Float32, num_dims, dim_size, repeat)
    println("=========== Data type is FP64 ==========")
    ok &= run_flip(Float64, num_dims, dim_size, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
