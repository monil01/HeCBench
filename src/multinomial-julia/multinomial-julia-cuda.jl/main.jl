using CUDA
using Printf
using Random

const GPU_NUM_THREADS = 256

function sample_multinomial_once_kernel!(dest, distributions::Int32, categories::Int32, sampled, dist, stride_dist::Int32, stride_categories::Int32)
    cur_dist = (blockIdx().x - Int32(1))
    stride_grid = gridDim().x
    while cur_dist < distributions
        if threadIdx().x == Int32(1)
            sum = 0.0f0
            for cat in Int32(0):(categories - Int32(1))
                @inbounds sum += dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)]
            end
            if sum == 0.0f0
                @inbounds dest[cur_dist + Int32(1)] = Int32(0)
            else
                sample = @inbounds sampled[cur_dist + Int32(1)]
                prev_bucket = 0.0f0
                found = false
                found_pos = Int32(0)
                for cat in Int32(0):(categories - Int32(1))
                    dist_val = @inbounds dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)] / sum
                    cur_bucket = prev_bucket + dist_val
                    if sample < cur_bucket && sample >= prev_bucket && dist_val > 0.0f0
                        found_pos = cat
                        found = true
                        break
                    end
                    prev_bucket = cur_bucket
                end
                if found
                    @inbounds dest[cur_dist + Int32(1)] = found_pos
                else
                    for cat in (categories - Int32(1)):-Int32(1):Int32(0)
                        if @inbounds dist[cur_dist * stride_dist + cat * stride_categories + Int32(1)] > 0.0f0
                            @inbounds dest[cur_dist + Int32(1)] = cat
                            break
                        end
                    end
                end
            end
        end
        cur_dist += stride_grid
    end
    return
end

function sample_multinomial_once_cpu(distributions, categories, sampled, dist, stride_dist, stride_categories)
    dest = Vector{Int32}(undef, distributions)
    for cur_dist in 0:(distributions - 1)
        sum = 0.0f0
        for cat in 0:(categories - 1)
            sum += dist[cur_dist * stride_dist + cat * stride_categories + 1]
        end
        if sum == 0.0f0
            dest[cur_dist + 1] = 0
            continue
        end
        sample = sampled[cur_dist + 1]
        prev_bucket = 0.0f0
        found = false
        found_pos = 0
        for cat in 0:(categories - 1)
            dist_val = dist[cur_dist * stride_dist + cat * stride_categories + 1] / sum
            cur_bucket = prev_bucket + dist_val
            if sample < cur_bucket && sample >= prev_bucket && dist_val > 0.0f0
                found_pos = cat
                found = true
                break
            end
            prev_bucket = cur_bucket
        end
        if found
            dest[cur_dist + 1] = Int32(found_pos)
        else
            for cat in (categories - 1):-1:0
                if dist[cur_dist * stride_dist + cat * stride_categories + 1] > 0.0f0
                    dest[cur_dist + 1] = Int32(cat)
                    break
                end
            end
        end
    end
    return dest
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of distributions> <number of categories> <repeat>")
        return 1
    end
    num_dist = parse(Int, args[1])
    num_categories = parse(Int, args[2])
    repeat = parse(Int, args[3])

    rng = MersenneTwister(123)
    sample = rand(rng, Float32, num_dist)
    distr_rng = MersenneTwister(123)
    distr = Float32.(rand(distr_rng, 1:100, num_dist * num_categories))

    result_ref = sample_multinomial_once_cpu(num_dist, num_categories, sample, distr, num_categories, 1)
    d_sample = CuArray(sample)
    d_distr = CuArray(distr)
    d_result = CUDA.zeros(Int32, num_dist)

    @cuda threads=GPU_NUM_THREADS blocks=512 sample_multinomial_once_kernel!(
        d_result, Int32(num_dist), Int32(num_categories), d_sample, d_distr, Int32(num_categories), Int32(1))
    result = Array(d_result)

    error = 0
    for i in 1:num_dist
        if abs(result[i] - result_ref[i]) > 1
            println("results mismatch: $(i - 1) $(result[i]) $(result_ref[i])")
            error = 1
            break
        end
    end
    println(error != 0 ? "FAIL" : "PASS")

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=GPU_NUM_THREADS blocks=512 sample_multinomial_once_kernel!(
            d_result, Int32(num_dist), Int32(num_categories), d_sample, d_distr, Int32(num_categories), Int32(1))
    end
    CUDA.synchronize()
    @printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)

    return error == 0 ? 0 : 1
end

exit(main(ARGS))
