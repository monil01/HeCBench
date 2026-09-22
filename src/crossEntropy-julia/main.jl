using CUDA
using Printf
using Random

const BS = 128
const W = 81
const H = 8732
const THREAD_X = 64
const THREAD_BS = 1
const TOLERANCE = 4.0f-3

function loss_bwd_kernel!(log_softmax, grad_output, grad_output_neg, target, weight,
                          mask, grad_predict, ::Type{T}) where {T}
    local_x = threadIdx().x - Int32(1)
    group_x = blockIdx().x - Int32(1)
    group_bs = blockIdx().y - Int32(1)
    linear_x = group_x * Int32(THREAD_X) + local_x
    if linear_x >= Int32(H)
        return
    end

    offset2d0 = group_bs * Int32(H) + linear_x
    idx = target[offset2d0 + Int32(1)]
    sum_offset0 = group_bs * Int32(W * H) + idx * Int32(H) + linear_x

    @inbounds begin
        tmp_grad = mask[offset2d0 + Int32(1)] != Int64(0) ?
                   -(grad_output[offset2d0 + Int32(1)] + grad_output_neg[offset2d0 + Int32(1)]) :
                   -grad_output[offset2d0 + Int32(1)]
        tmp_grad *= weight[offset2d0 + Int32(1)]
        sum_value = Float32(tmp_grad) * Float32(log_softmax[sum_offset0 + Int32(1)])

        i = Int32(0)
        while i < Int32(W)
            in_offset0 = group_bs * Int32(W * H) + i * Int32(H) + linear_x
            tmp_sfm = exp(Float32(log_softmax[in_offset0 + Int32(1)])) * sum_value
            res = (i == idx) ? Float32(tmp_grad) - tmp_sfm : -tmp_sfm
            grad_predict[in_offset0 + Int32(1)] = T(res)
            i += Int32(1)
        end
    end
    return
end

function loss_bwd_cpu!(grad_predict::Vector{T}, log_softmax::Vector{T}, target,
                       weight::Vector{T}, mask, grad_output::Vector{T},
                       grad_output_neg::Vector{T}) where {T}
    fill!(grad_predict, zero(T))
    sum_value = Vector{Float32}(undef, BS * H)

    @inbounds for b in 0:BS-1
        for j in 0:H-1
            off2 = b * H + j + 1
            idx = Int(target[off2])
            pred_off = b * W * H + idx * H + j + 1
            tmp = -(Float32(grad_output[off2]) +
                    (mask[off2] != 0 ? Float32(grad_output_neg[off2]) : 0.0f0)) *
                  Float32(weight[off2])
            grad_predict[pred_off] = T(tmp)
            sum_value[b * H + j + 1] = tmp * Float32(log_softmax[pred_off])
        end
    end

    @inbounds for b in 0:BS-1
        for k in 0:W-1
            for j in 0:H-1
                off = b * W * H + k * H + j + 1
                grad_predict[off] = T(Float32(grad_predict[off]) -
                                      exp(Float32(log_softmax[off])) *
                                      sum_value[b * H + j + 1])
            end
        end
    end
    return grad_predict
end

function run_type(::Type{T}, repeat::Int) where {T}
    predict_shape = BS * W * H
    output_shape = BS * H
    rng = MersenneTwister(1234 + sizeof(T))

    log_softmax = T.(rand(rng, Float32, predict_shape))
    grad_output = T.(rand(rng, Float32, output_shape))
    grad_output_neg = T.(rand(rng, Float32, output_shape))
    weight = T.(rand(rng, Float32, output_shape))
    target = Int64.(rand(rng, 0:W-2, output_shape))
    mask = Int64.(rand(rng, 0:1, output_shape))
    grad_predict_ref = Vector{T}(undef, predict_shape)

    CUDA.synchronize()
    t_alloc0 = time_ns()
    d_log_softmax = CuArray(log_softmax)
    d_grad_output = CuArray(grad_output)
    d_grad_output_neg = CuArray(grad_output_neg)
    d_weight = CuArray(weight)
    d_target = CuArray(target)
    d_mask = CuArray(mask)
    d_grad_predict = CUDA.zeros(T, predict_shape)
    CUDA.synchronize()
    transfer_ms = (time_ns() - t_alloc0) * 1.0e-6

    blocks = (cld(H, THREAD_X), BS)
    warmup = 10
    kernel_ms = 0.0
    for k in 1:(warmup + repeat)
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=(THREAD_X, THREAD_BS) blocks=blocks loss_bwd_kernel!(
            d_log_softmax, d_grad_output, d_grad_output_neg, d_target, d_weight,
            d_mask, d_grad_predict, T)
        CUDA.synchronize()
        if k > warmup
            kernel_ms += (time_ns() - t0) * 1.0e-6
        end
    end

    t_copy0 = time_ns()
    grad_predict_gpu = Array(d_grad_predict)
    CUDA.synchronize()
    transfer_ms += (time_ns() - t_copy0) * 1.0e-6

    t_cpu0 = time_ns()
    loss_bwd_cpu!(grad_predict_ref, log_softmax, target, weight, mask,
                  grad_output, grad_output_neg)
    cpu_ms = (time_ns() - t_cpu0) * 1.0e-6

    errors = 0
    @inbounds for i in eachindex(grad_predict_ref)
        if abs(Float32(grad_predict_ref[i]) - Float32(grad_predict_gpu[i])) > TOLERANCE
            errors += 1
            if errors < 10
                @printf("Error, output not equal, i=%d, cpu_result = %f, device_result = %f, gap = %f\n",
                        i - 1, Float32(grad_predict_ref[i]), Float32(grad_predict_gpu[i]),
                        Float32(grad_predict_ref[i]) - Float32(grad_predict_gpu[i]))
            end
        end
    end

    @printf("GPU device memory allocation and data transfer time (ms) : %f\n", transfer_ms / repeat)
    @printf("Average GPU kernel time (ms) : %f\n", kernel_ms / repeat)
    @printf("CPU serial time (ms) : %f\n", cpu_ms)
    bytes = Float64(sizeof(T)) * Float64(predict_shape * 2 + output_shape * 3) +
            Float64(sizeof(Int64)) * Float64(output_shape * 2)
    @printf("BandWidth = %f (GB / s)\n", bytes / ((kernel_ms / repeat) / 1000.0) / 1.0e9)
    return errors == 0
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    @printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d \n", BS, W, H)

    all_ok = true
    println("=========== Data type is FP16 ==========")
    all_ok &= run_type(Float16, repeat)
    println(all_ok ? "PASS" : "FAIL")
    GC.gc(true)

    println("=========== Data type is FP32 ==========")
    all_ok &= run_type(Float32, repeat)
    println(all_ok ? "PASS" : "FAIL")
    GC.gc(true)

    println("=========== Data type is FP64 ==========")
    all_ok &= run_type(Float64, repeat)
    println(all_ok ? "PASS" : "FAIL")
    return all_ok ? 0 : 1
end

exit(main(ARGS))
