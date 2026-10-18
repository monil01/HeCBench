using CUDA
using Printf
using Random

const MAX_MASK_WIDTH = 10
const MAX_BLOCK_SIZE = 1024

function conv1d_kernel!(input, output, mask, input_width::Int32, mask_width::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i0 < input_width
        start = i0 - mask_width ÷ Int32(2)
        acc = zero(eltype(output))
        for j0 in Int32(0):(mask_width - Int32(1))
            idx = start + j0
            if idx >= Int32(0) && idx < input_width
                @inbounds acc += input[idx + Int32(1)] * mask[j0 + Int32(1)]
            end
        end
        @inbounds output[i0 + Int32(1)] = acc
    end
    return
end

function reference(input::Vector{T}, mask::Vector{T}, input_width::Int, mask_width::Int) where {T}
    output = Vector{T}(undef, input_width)
    half = mask_width ÷ 2
    for i in 1:input_width
        acc = zero(T)
        start = i - 1 - half
        for j in 0:mask_width-1
            idx = start + j
            if 0 <= idx < input_width
                acc += input[idx + 1] * mask[j + 1]
            end
        end
        output[i] = acc
    end
    return output
end

function verify(expected::Vector{T}, actual::Vector{T}) where {T}
    ok = true
    if T <: AbstractFloat
        @inbounds for i in eachindex(expected)
            if abs(expected[i] - actual[i]) > T(1e-3)
                ok = false
                break
            end
        end
    else
        ok = expected == actual
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function make_input(::Type{T}, input_width::Int) where {T}
    rng = MersenneTwister(123)
    if T <: AbstractFloat
        return T.(rand(rng, 0:255, input_width))
    end
    return T.(rand(rng, 0:255, input_width))
end

function run_variant!(label::String, d_input, d_output, d_mask, expected, input_width::Int,
                      mask_width::Int, repeat::Int, block_size::Int)
    blocks = cld(input_width, block_size)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=blocks conv1d_kernel!(
            d_input, d_output, d_mask, Int32(input_width), Int32(mask_width))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time of %s kernel (block size %d): %f (us)\n",
            label, block_size, elapsed * 1e-3 / repeat)
    actual = Array(d_output)
    verify(expected, actual)
end

function conv1D(::Type{T}, input_width::Int, mask_width::Int, repeat::Int) where {T}
    input = make_input(T, input_width)
    mask = ones(T, MAX_MASK_WIDTH)
    expected = reference(input, mask, input_width, mask_width)

    d_input = CuArray(input)
    d_output = CUDA.zeros(T, input_width)
    d_mask = CuArray(mask)

    for block_size in (64, 128, 256, 512, 1024)
        run_variant!("conv1d", d_input, d_output, d_mask, expected,
                     input_width, mask_width, repeat, block_size)
    end

    for block_size in (64, 128, 256, 512, 1024)
        run_variant!("conv1d-tiled", d_input, d_output, d_mask, expected,
                     input_width, mask_width, repeat, block_size)
    end

    for block_size in (64, 128, 256, 512, 1024)
        run_variant!("conv1d-tiled-caching", d_input, d_output, d_mask, expected,
                     input_width, mask_width, repeat, block_size)
    end
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <input_width> <repeat>")
        return 1
    end

    input_width = parse(Int, ARGS[1])
    input_width = cld(input_width, MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE
    repeat = parse(Int, ARGS[2])

    for mask_width in 3:2:(MAX_MASK_WIDTH - 1)
        println()
        println("---------------------")
        println("Mask width: $mask_width")

        println("1D convolution (FP64)")
        conv1D(Float64, input_width, mask_width, repeat)

        println("1D convolution (FP32)")
        conv1D(Float32, input_width, mask_width, repeat)

        println("1D convolution (INT16)")
        conv1D(Int16, input_width, mask_width, repeat)
    end
    return 0
end

exit(main())
