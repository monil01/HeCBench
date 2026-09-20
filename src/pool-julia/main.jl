using CUDA
using Printf
using Random

const BSIZE = 256
const KSIZE_H = Int32(11)
const KSIZE_W = Int32(11)
const STRIDE_H = Int32(4)
const STRIDE_W = Int32(4)
const PAD_H = Int32(1)
const PAD_W = Int32(1)

function pool2d_grad_kernel!(nthreads::Int32, input_data, output_data, output_grad,
                             channels::Int32, input_h::Int32, input_w::Int32,
                             output_h::Int32, output_w::Int32, input_grad)
    index0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while index0 < nthreads
        w_offset = index0 % input_w + PAD_W
        tmp = index0 ÷ input_w
        h_offset = tmp % input_h + PAD_H
        tmp ÷= input_h
        offset_c = tmp % channels
        batch_idx = tmp ÷ channels

        phstart = h_offset < KSIZE_H ? Int32(0) : (h_offset - KSIZE_H) ÷ STRIDE_H + Int32(1)
        pwstart = w_offset < KSIZE_W ? Int32(0) : (w_offset - KSIZE_W) ÷ STRIDE_W + Int32(1)
        phend = min(h_offset ÷ STRIDE_H + Int32(1), output_h)
        pwend = min(w_offset ÷ STRIDE_W + Int32(1), output_w)

        gradient = 0.0f0
        output_stride = batch_idx * output_h * output_w * channels +
                        offset_c * output_h * output_w
        for ph in phstart:phend-Int32(1)
            for pw in pwstart:pwend-Int32(1)
                hstart = ph * STRIDE_H - PAD_H
                wstart = pw * STRIDE_W - PAD_W
                hend = min(hstart + KSIZE_H, input_h)
                wend = min(wstart + KSIZE_W, input_w)
                hstart = max(hstart, Int32(0))
                wstart = max(wstart, Int32(0))
                pool_size = (hend - hstart) * (wend - wstart)
                output_sub_idx = ph * output_w + pw
                @inbounds gradient += (1.0f0 / Float32(pool_size)) *
                    output_grad[output_stride + output_sub_idx + Int32(1)]
            end
        end
        @inbounds input_grad[index0 + Int32(1)] = gradient
        index0 += stride
    end
    return
end

function reference!(nthreads::Int, output_grad, channels::Int, input_h::Int, input_w::Int,
                    output_h::Int, output_w::Int, input_grad)
    for index0 in 0:nthreads-1
        w_offset = index0 % input_w + Int(PAD_W)
        tmp = index0 ÷ input_w
        h_offset = tmp % input_h + Int(PAD_H)
        tmp ÷= input_h
        offset_c = tmp % channels
        batch_idx = tmp ÷ channels

        phstart = h_offset < Int(KSIZE_H) ? 0 : (h_offset - Int(KSIZE_H)) ÷ Int(STRIDE_H) + 1
        pwstart = w_offset < Int(KSIZE_W) ? 0 : (w_offset - Int(KSIZE_W)) ÷ Int(STRIDE_W) + 1
        phend = min(h_offset ÷ Int(STRIDE_H) + 1, output_h)
        pwend = min(w_offset ÷ Int(STRIDE_W) + 1, output_w)

        gradient = 0.0f0
        output_stride = batch_idx * output_h * output_w * channels +
                        offset_c * output_h * output_w
        for ph in phstart:phend-1, pw in pwstart:pwend-1
            hstart = ph * Int(STRIDE_H) - Int(PAD_H)
            wstart = pw * Int(STRIDE_W) - Int(PAD_W)
            hend = min(hstart + Int(KSIZE_H), input_h)
            wend = min(wstart + Int(KSIZE_W), input_w)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            pool_size = (hend - hstart) * (wend - wstart)
            output_sub_idx = ph * output_w + pw
            gradient += (1.0f0 / Float32(pool_size)) * output_grad[output_stride + output_sub_idx + 1]
        end
        input_grad[index0 + 1] = gradient
    end
end

function main(args)
    if length(args) != 7
        println("Usage: main.jl <batch> <input channels> <input height> <input width> <output height> <output width> <repeat>")
        return 1
    end
    batch = parse(Int, args[1])
    channels = parse(Int, args[2])
    input_h = parse(Int, args[3])
    input_w = parse(Int, args[4])
    output_h = parse(Int, args[5])
    output_w = parse(Int, args[6])
    repeat = parse(Int, args[7])

    input_numel = batch * channels * input_h * input_w
    output_numel = batch * channels * output_h * output_w
    nthreads = input_numel

    rng = MersenneTwister(123)
    input = rand(rng, Float32, input_numel)
    output = rand(rng, Float32, output_numel)
    output_grad = fill(Float32(input_w * input_h), output_numel)
    input_grad_ref = Vector{Float32}(undef, input_numel)

    d_input = CuArray(input)
    d_output = CuArray(output)
    d_output_grad = CuArray(output_grad)
    d_input_grad = CUDA.zeros(Float32, input_numel)
    blocks = cld(nthreads, BSIZE)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BSIZE blocks=blocks pool2d_grad_kernel!(
            Int32(nthreads), d_input, d_output, d_output_grad,
            Int32(channels), Int32(input_h), Int32(input_w),
            Int32(output_h), Int32(output_w), d_input_grad)
    end
    CUDA.synchronize()
    @printf("Average kernel execution time: %f (s)\n", (time_ns() - t0) * 1.0e-9 / repeat)

    input_grad = Array(d_input_grad)
    reference!(nthreads, output_grad, channels, input_h, input_w, output_h, output_w, input_grad_ref)
    ok = all(abs.(input_grad .- input_grad_ref) .<= 1.0f-3)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
