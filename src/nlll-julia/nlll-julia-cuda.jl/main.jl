using CUDA
using Printf
using Random

function nll_loss_kernel!(output, total_weight, input, target, weights,
                          size_average::Bool, nframe::Int64, kdim::Int64,
                          ignore_index::Int64, ::Val{N}) where {N}
    tid = threadIdx().x
    sm_inputs = @cuDynamicSharedMem(Float32, N)
    sm_weights = @cuDynamicSharedMem(Float32, N, N * sizeof(Float32))
    acc = 0.0f0
    wacc = 0.0f0
    i = Int64(tid) - 1
    while i < nframe
        t = Int64(@inbounds target[i + 1])
        if t != ignore_index
            cur_weight = Float32(@inbounds weights[t + 1])
            acc -= Float32(@inbounds input[i * kdim + t + 1]) * cur_weight
            wacc += cur_weight
        end
        i += N
    end
    @inbounds sm_inputs[tid] = acc
    @inbounds sm_weights[tid] = wacc
    sync_threads()
    if tid == 1
        out = 0.0f0
        wt = 0.0f0
        for j in 1:N
            out += @inbounds sm_inputs[j]
            wt += @inbounds sm_weights[j]
        end
        @inbounds total_weight[1] = eltype(total_weight)(wt)
        @inbounds output[1] = eltype(output)(size_average ? out / wt : out)
    end
    return
end

function reference(input, target, weights, size_average::Bool, nframe::Int, kdim::Int, ignore_index::Int, ::Type{T}) where {T}
    out = 0.0f0
    wt = 0.0f0
    @inbounds for i in 0:nframe-1
        t = Int(target[i + 1])
        if t != ignore_index
            cur_weight = Float32(weights[t + 1])
            out -= Float32(input[i * kdim + t + 1]) * cur_weight
            wt += cur_weight
        end
    end
    return T(size_average ? out / wt : out), T(wt)
end

function run_case(::Type{T}, nframe::Int, nclasses::Int, repeat::Int) where {T}
    input_size = nframe * nclasses
    rng = MersenneTwister(123)
    input = T.(rand(rng, Float32, input_size) .* 2.0f0 .- 1.0f0)
    weights = T.(rand(rng, Float32, nframe) .* 2.0f0 .- 1.0f0)
    target = Int32.(rand(rng, 0:nclasses-1, nframe))
    size_average = true
    ignore_index = nclasses ÷ 2
    ref_output, ref_weight = reference(input, target, weights, size_average, nframe, nclasses, ignore_index, T)

    d_input = CuArray(input)
    d_weights = CuArray(weights)
    d_target = CuArray(target)
    d_output = CUDA.zeros(T, 1)
    d_total_weight = CUDA.zeros(T, 1)
    ok_all = true
    for threads in (64, 128, 256, 512, 1024)
        shmem = 2 * threads * sizeof(Float32)
        CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:repeat
            @cuda threads=threads blocks=1 shmem=shmem nll_loss_kernel!(
                d_output, d_total_weight, d_input, d_target, d_weights,
                size_average, Int64(nframe), Int64(nclasses), Int64(ignore_index), Val(threads))
        end
        CUDA.synchronize()
        println()
        @printf("Thread block size: %d\n", threads)
        @printf("Average execution time of nll loss forward kernel: %f (us)\n",
                (time_ns() - t0) * 1.0e-3 / repeat)
        h_output = Array(d_output)[1]
        h_weight = Array(d_total_weight)[1]
        ok = abs(Float32(h_output) - Float32(ref_output)) <= 1.0f0 &&
             abs(Float32(h_weight) - Float32(ref_weight)) <= 1.0f0
        if !ok
            @printf("%f %f %f %f\n", Float32(h_output), Float32(ref_output),
                    Float32(h_weight), Float32(ref_weight))
        end
        println(ok ? "PASS" : "FAIL")
        ok_all &= ok
    end
    return ok_all
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <minibatch size> <number of classes> <repeat>")
        return 1
    end
    nframe = parse(Int, args[1])
    nclasses = parse(Int, args[2])
    repeat = parse(Int, args[3])

    println("=========== Data type is FP32 ==========")
    ok32 = run_case(Float32, nframe, nclasses, repeat)
    println("=========== Data type is FP16 ==========")
    ok16 = run_case(Float16, nframe, nclasses, repeat)
    return (ok32 && ok16) ? 0 : 1
end

exit(main(ARGS))
