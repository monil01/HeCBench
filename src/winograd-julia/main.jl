using CUDA
using Printf
using StaticArrays

const MAP_SIZE = 1024
const OUT_SIZE = MAP_SIZE - 2
const TILE_N = (OUT_SIZE + 1) ÷ 2
const DIM_LOCAL_WORK_GROUP_X = 32
const DIM_LOCAL_WORK_GROUP_Y = 8
const SMALL_FLOAT_VAL = Float32(1.0f-8)
const PERCENT_DIFF_ERROR_THRESHOLD = Float32(1.05)
const RAND_MAX_C = Float32(2147483647)

function rand_float32()
    return Float32(ccall(:rand, Cint, ())) / RAND_MAX_C
end

function filter_transformation()
    filter = @SMatrix Float32[
        0.2 -0.3 0.4;
        0.5 0.6 0.7;
       -0.8 -0.9 0.10
    ]
    tmp = MMatrix{4, 3, Float32}(undef)
    @inbounds for j in 1:3
        tmp[1, j] = filter[1, j]
        tmp[2, j] = Float32(0.5) * filter[1, j] + Float32(0.5) * filter[2, j] + Float32(0.5) * filter[3, j]
        tmp[3, j] = Float32(0.5) * filter[1, j] - Float32(0.5) * filter[2, j] + Float32(0.5) * filter[3, j]
        tmp[4, j] = filter[3, j]
    end

    transformed = Matrix{Float32}(undef, 4, 4)
    @inbounds for i in 1:4
        transformed[i, 1] = tmp[i, 1]
        transformed[i, 2] = Float32(0.5) * tmp[i, 1] + Float32(0.5) * tmp[i, 2] + Float32(0.5) * tmp[i, 3]
        transformed[i, 3] = Float32(0.5) * tmp[i, 1] - Float32(0.5) * tmp[i, 2] + Float32(0.5) * tmp[i, 3]
        transformed[i, 4] = tmp[i, 3]
    end
    return vec(transformed')
end

@inline function winograd_tile(input, transformed_filter, tile_i::Int, tile_j::Int)
    input_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile = MMatrix{4, 4, Float32}(undef)
    transformed_tile = MMatrix{4, 4, Float32}(undef)
    multiplied_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile_1 = MMatrix{2, 4, Float32}(undef)
    final_tile = MMatrix{2, 2, Float32}(undef)

    @inbounds for i in 0:3
        for j in 0:3
            x = 2 * tile_i + i
            y = 2 * tile_j + j
            input_tile[i + 1, j + 1] = (x >= MAP_SIZE || y >= MAP_SIZE) ? Float32(0) : input[x + 1, y + 1]
        end
    end

    @inbounds for j in 1:4
        tmp_tile[1, j] = input_tile[1, j] - input_tile[3, j]
        tmp_tile[2, j] = input_tile[2, j] + input_tile[3, j]
        tmp_tile[3, j] = -input_tile[2, j] + input_tile[3, j]
        tmp_tile[4, j] = input_tile[2, j] - input_tile[4, j]
    end
    @inbounds for i in 1:4
        transformed_tile[i, 1] = tmp_tile[i, 1] - tmp_tile[i, 3]
        transformed_tile[i, 2] = tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 3] = -tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 4] = tmp_tile[i, 2] - tmp_tile[i, 4]
    end

    @inbounds for i in 1:4
        for j in 1:4
            multiplied_tile[i, j] = transformed_tile[i, j] * transformed_filter[(i - 1) * 4 + j]
        end
    end

    @inbounds for j in 1:4
        tmp_tile_1[1, j] = multiplied_tile[1, j] + multiplied_tile[2, j] + multiplied_tile[3, j]
        tmp_tile_1[2, j] = multiplied_tile[2, j] - multiplied_tile[3, j] - multiplied_tile[4, j]
    end
    @inbounds for i in 1:2
        final_tile[i, 1] = tmp_tile_1[i, 1] + tmp_tile_1[i, 2] + tmp_tile_1[i, 3]
        final_tile[i, 2] = tmp_tile_1[i, 2] - tmp_tile_1[i, 3] - tmp_tile_1[i, 4]
    end
    return final_tile
end

function winograd_cpu!(input, output, transformed_filter, tile_i_stop::Int=TILE_N)
    fill!(output, Float32(0))
    @inbounds for tile_i in 0:(tile_i_stop - 1)
        for tile_j in 0:(TILE_N - 1)
            final_tile = winograd_tile(input, transformed_filter, tile_i, tile_j)
            for i in 0:1
                for j in 0:1
                    x = 2 * tile_i + i
                    y = 2 * tile_j + j
                    if x < OUT_SIZE && y < OUT_SIZE
                        output[x + 1, y + 1] = final_tile[i + 1, j + 1]
                    end
                end
            end
        end
    end
    return output
end

function percent_diff(a::Float32, b::Float32)
    if abs(a) < Float32(0.01) && abs(b) < Float32(0.01)
        return Float32(0)
    end
    return Float32(100) * abs(abs(a - b) / abs(a + SMALL_FLOAT_VAL))
end

function compare_results(reference, actual)
    @inbounds for j in axes(reference, 2)
        for i in axes(reference, 1)
            if percent_diff(reference[i, j], actual[i, j]) > PERCENT_DIFF_ERROR_THRESHOLD
                return false
            end
        end
    end
    return true
end

function winograd_kernel!(input, transformed_filter, output, offset_i::Int32, offset_j::Int32)
    tile_i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1) + offset_i
    tile_j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1) + offset_j

    input_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile = MMatrix{4, 4, Float32}(undef)
    transformed_tile = MMatrix{4, 4, Float32}(undef)
    multiplied_tile = MMatrix{4, 4, Float32}(undef)
    tmp_tile_1 = MMatrix{2, 4, Float32}(undef)
    final_tile = MMatrix{2, 2, Float32}(undef)

    @inbounds for i in Int32(0):Int32(3)
        for j in Int32(0):Int32(3)
            x = Int32(2) * tile_i + i
            y = Int32(2) * tile_j + j
            input_tile[Int(i) + 1, Int(j) + 1] =
                (x >= Int32(MAP_SIZE) || y >= Int32(MAP_SIZE)) ? Float32(0) : input[x + Int32(1), y + Int32(1)]
        end
    end

    @inbounds for j in 1:4
        tmp_tile[1, j] = input_tile[1, j] - input_tile[3, j]
        tmp_tile[2, j] = input_tile[2, j] + input_tile[3, j]
        tmp_tile[3, j] = -input_tile[2, j] + input_tile[3, j]
        tmp_tile[4, j] = input_tile[2, j] - input_tile[4, j]
    end
    @inbounds for i in 1:4
        transformed_tile[i, 1] = tmp_tile[i, 1] - tmp_tile[i, 3]
        transformed_tile[i, 2] = tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 3] = -tmp_tile[i, 2] + tmp_tile[i, 3]
        transformed_tile[i, 4] = tmp_tile[i, 2] - tmp_tile[i, 4]
    end

    @inbounds for i in 1:4
        for j in 1:4
            multiplied_tile[i, j] = transformed_tile[i, j] * transformed_filter[(i - 1) * 4 + j]
        end
    end

    @inbounds for j in 1:4
        tmp_tile_1[1, j] = multiplied_tile[1, j] + multiplied_tile[2, j] + multiplied_tile[3, j]
        tmp_tile_1[2, j] = multiplied_tile[2, j] - multiplied_tile[3, j] - multiplied_tile[4, j]
    end
    @inbounds for i in 1:2
        final_tile[i, 1] = tmp_tile_1[i, 1] + tmp_tile_1[i, 2] + tmp_tile_1[i, 3]
        final_tile[i, 2] = tmp_tile_1[i, 2] - tmp_tile_1[i, 3] - tmp_tile_1[i, 4]
    end

    @inbounds for i in Int32(0):Int32(1)
        for j in Int32(0):Int32(1)
            x = Int32(2) * tile_i + i
            y = Int32(2) * tile_j + j
            if x < Int32(OUT_SIZE) && y < Int32(OUT_SIZE)
                output[x + Int32(1), y + Int32(1)] = final_tile[Int(i) + 1, Int(j) + 1]
            end
        end
    end
    return
end

function main()
    start_total = time_ns()
    ccall(:srand, Cvoid, (Cuint,), UInt32(1))
    input = Matrix{Float32}(undef, MAP_SIZE, MAP_SIZE)
    @inbounds for j in 1:MAP_SIZE
        for i in 1:MAP_SIZE
            input[i, j] = rand_float32()
        end
    end

    transformed_filter = filter_transformation()
    reference = Matrix{Float32}(undef, OUT_SIZE, OUT_SIZE)
    winograd_cpu!(input, reference, transformed_filter)

    d_input = CuArray(input)
    d_filter = CuArray(transformed_filter)
    d_output = CUDA.zeros(Float32, OUT_SIZE, OUT_SIZE)
    cpu_output = Matrix{Float32}(undef, OUT_SIZE, OUT_SIZE)

    global_x = cld(TILE_N, DIM_LOCAL_WORK_GROUP_X) * DIM_LOCAL_WORK_GROUP_X
    global_y = cld(TILE_N, DIM_LOCAL_WORK_GROUP_Y) * DIM_LOCAL_WORK_GROUP_Y

    pass = true
    co_time_ns = 0
    for cpu_offset in 0:100
        cpu_global_x = (cpu_offset * cld(TILE_N, DIM_LOCAL_WORK_GROUP_X) ÷ 100) * DIM_LOCAL_WORK_GROUP_X
        gpu_global_x = global_x - cpu_global_x
        gpu_run = gpu_global_x > 0
        cpu_run = cpu_global_x > 0

        co_start = time_ns()
        if gpu_run
            @cuda threads=(DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y) blocks=(gpu_global_x ÷ DIM_LOCAL_WORK_GROUP_X, global_y ÷ DIM_LOCAL_WORK_GROUP_Y) winograd_kernel!(
                d_input, d_filter, d_output, Int32(cpu_global_x), Int32(0))
        end

        if cpu_run
            winograd_cpu!(input, cpu_output, transformed_filter, cpu_global_x)
            rows_to_copy = gpu_run ? min(cpu_global_x * 2, OUT_SIZE) : OUT_SIZE
            d_output[1:rows_to_copy, :] = cpu_output[1:rows_to_copy, :]
        end

        actual = Array(d_output)
        co_time_ns += time_ns() - co_start
        pass &= compare_results(reference, actual)
    end

    println(pass ? "PASS" : "FAIL")
    total_time_ns = time_ns() - start_total
    @printf("Co-execution time: %lf s\n", co_time_ns * 1e-9)
    @printf("Total time: %lf s\n", total_time_ns * 1e-9)
    @printf("Ratio of co-execution time to total time: %.2lf%%\n", 100.0 * co_time_ns / total_time_ns)
end

main()
