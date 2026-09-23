using CUDA
using Printf
using Random

function cal_num_less_kernel(x::Float32, width::UInt32, diagonal, off_diagonal)
    count = UInt32(0)
    prev_diff = diagonal[1] - x
    count += prev_diff < 0f0 ? UInt32(1) : UInt32(0)
    i = UInt32(2)
    while i <= width
        diff = (diagonal[Int(i)] - x) -
               (off_diagonal[Int(i - UInt32(1))] * off_diagonal[Int(i - UInt32(1))]) / prev_diff
        count += diff < 0f0 ? UInt32(1) : UInt32(0)
        prev_diff = diff
        i += UInt32(1)
    end
    return count
end

function cal_num_eigenvalue_interval_kernel!(num_intervals, eigen_intervals,
                                             diagonal, off_diagonal,
                                             width::UInt32)
    gid0 = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if gid0 >= width
        return
    end
    lower_id = UInt32(2) * gid0
    upper_id = lower_id + UInt32(1)
    lower_limit = eigen_intervals[Int(lower_id) + 1]
    upper_limit = eigen_intervals[Int(upper_id) + 1]
    lower = cal_num_less_kernel(lower_limit, width, diagonal, off_diagonal)
    upper = cal_num_less_kernel(upper_limit, width, diagonal, off_diagonal)
    num_intervals[Int(gid0) + 1] = upper - lower
    return
end

function recalculate_eigen_intervals_kernel!(new_intervals, eigen_intervals,
                                             num_intervals, diagonal,
                                             off_diagonal, width::UInt32,
                                             tolerance::Float32)
    gid0 = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if gid0 >= width
        return
    end
    lower_id = UInt32(2) * gid0
    upper_id = lower_id + UInt32(1)
    current_index = gid0

    index = UInt32(0)
    while current_index >= num_intervals[Int(index) + 1]
        current_index -= num_intervals[Int(index) + 1]
        index += UInt32(1)
    end

    l_id = UInt32(2) * index
    u_id = l_id + UInt32(1)
    interval_count = num_intervals[Int(index) + 1]

    if interval_count == UInt32(1)
        mid_value = (eigen_intervals[Int(u_id) + 1] + eigen_intervals[Int(l_id) + 1]) / 2f0
        n = cal_num_less_kernel(mid_value, width, diagonal, off_diagonal) -
            cal_num_less_kernel(eigen_intervals[Int(l_id) + 1], width, diagonal, off_diagonal)

        if eigen_intervals[Int(u_id) + 1] - eigen_intervals[Int(l_id) + 1] < tolerance
            new_intervals[Int(lower_id) + 1] = eigen_intervals[Int(l_id) + 1]
            new_intervals[Int(upper_id) + 1] = eigen_intervals[Int(u_id) + 1]
        elseif n == UInt32(0)
            new_intervals[Int(lower_id) + 1] = mid_value
            new_intervals[Int(upper_id) + 1] = eigen_intervals[Int(u_id) + 1]
        else
            new_intervals[Int(lower_id) + 1] = eigen_intervals[Int(l_id) + 1]
            new_intervals[Int(upper_id) + 1] = mid_value
        end
    else
        division_width = (eigen_intervals[Int(u_id) + 1] - eigen_intervals[Int(l_id) + 1]) / Float32(interval_count)
        new_intervals[Int(lower_id) + 1] = eigen_intervals[Int(l_id) + 1] + division_width * Float32(current_index)
        new_intervals[Int(upper_id) + 1] = new_intervals[Int(lower_id) + 1] + division_width
    end
    return
end

function cal_num_less_cpu(diagonal, off_diagonal, length::Int, x::Float32)
    count = UInt32(0)
    prev_diff = diagonal[1] - x
    count += prev_diff < 0f0 ? UInt32(1) : UInt32(0)
    for i in 2:length
        diff = (diagonal[i] - x) - ((off_diagonal[i - 1] * off_diagonal[i - 1]) / prev_diff)
        count += diff < 0f0 ? UInt32(1) : UInt32(0)
        prev_diff = diff
    end
    return count
end

function eigenvalue_cpu_reference!(diagonal, off_diagonal, length::Int,
                                   eigen_intervals, new_intervals,
                                   tolerance::Float32)
    offset = UInt32(0)
    for i0 in UInt32(0):UInt32(length - 1)
        lid = UInt32(2) * i0
        uid = lid + UInt32(1)
        less_lower = cal_num_less_cpu(diagonal, off_diagonal, length, eigen_intervals[Int(lid) + 1])
        less_upper = cal_num_less_cpu(diagonal, off_diagonal, length, eigen_intervals[Int(uid) + 1])
        num_sub = less_upper - less_lower

        if num_sub > UInt32(1)
            avg_width = (eigen_intervals[Int(uid) + 1] - eigen_intervals[Int(lid) + 1]) / Float32(num_sub)
            for j in UInt32(0):(num_sub - UInt32(1))
                new_lid = UInt32(2) * (offset + j)
                new_uid = new_lid + UInt32(1)
                new_intervals[Int(new_lid) + 1] = eigen_intervals[Int(lid) + 1] + Float32(j) * avg_width
                new_intervals[Int(new_uid) + 1] = new_intervals[Int(new_lid) + 1] + avg_width
            end
        elseif num_sub == UInt32(1)
            lower = eigen_intervals[Int(lid) + 1]
            upper = eigen_intervals[Int(uid) + 1]
            mid = (lower + upper) / 2f0
            new_lid = UInt32(2) * offset
            new_uid = new_lid + UInt32(1)
            if upper - lower < tolerance
                new_intervals[Int(new_lid) + 1] = lower
                new_intervals[Int(new_uid) + 1] = upper
            elseif cal_num_less_cpu(diagonal, off_diagonal, length, mid) == less_upper
                new_intervals[Int(new_lid) + 1] = lower
                new_intervals[Int(new_uid) + 1] = mid
            else
                new_intervals[Int(new_lid) + 1] = mid
                new_intervals[Int(new_uid) + 1] = upper
            end
        end
        offset += num_sub
    end
    return offset
end

function is_complete(eigen_intervals, length::Int, tolerance::Float32)
    for i in 0:(length - 1)
        lid = 2 * i + 1
        uid = lid + 1
        if eigen_intervals[uid] - eigen_intervals[lid] >= tolerance
            return true
        end
    end
    return false
end

function compute_gerschgorin(diagonal, off_diagonal, length::Int)
    lower = diagonal[1] - abs(off_diagonal[1])
    upper = diagonal[1] + abs(off_diagonal[1])
    for i in 2:(length - 1)
        r = abs(off_diagonal[i - 1]) + abs(off_diagonal[i])
        lower = min(lower, diagonal[i] - r)
        upper = max(upper, diagonal[i] + r)
    end
    lower = min(lower, diagonal[length] - abs(off_diagonal[length - 1]))
    upper = max(upper, diagonal[length] + abs(off_diagonal[length - 1]))
    return Float32(lower), Float32(upper)
end

function compare_intervals(ref_data, data; epsilon=1f-6)
    err = 0f0
    ref = 0f0
    for i in 2:length(ref_data)
        diff = ref_data[i] - data[i]
        err += diff * diff
        ref += ref_data[i] * ref_data[i]
    end
    if abs(ref) < 1f-7
        return false
    end
    return sqrt(err) / sqrt(ref) < epsilon
end

function round_to_power2(val::Int)
    v = val - 1
    shift = 1
    while shift < sizeof(Int) * 8
        v |= v >> shift
        shift <<= 1
    end
    return v + 1
end

function run_kernels!(d_diagonal, d_num_intervals, d_off_diagonal,
                      d_buffers, h_intervals, length::Int,
                      tolerance::Float32)
    for i in 1:2
        copyto!(d_buffers[i], h_intervals[i])
    end
    in_idx = 1
    grids = div(length, 256)
    while is_complete(h_intervals[in_idx], length, tolerance)
        @cuda threads=256 blocks=grids cal_num_eigenvalue_interval_kernel!(
            d_num_intervals, d_buffers[in_idx], d_diagonal, d_off_diagonal, UInt32(length))
        out_idx = 3 - in_idx
        @cuda threads=256 blocks=grids recalculate_eigen_intervals_kernel!(
            d_buffers[out_idx], d_buffers[in_idx], d_num_intervals, d_diagonal,
            d_off_diagonal, UInt32(length), tolerance)
        in_idx = out_idx
        copyto!(h_intervals[in_idx], d_buffers[in_idx])
    end
    return in_idx
end

function main()
    if length(ARGS) != 2
        println("Usage: main.jl <length of the diagonal of the square matrix> <repeat>")
        exit(1)
    end

    length_arg = parse(Int, ARGS[1])
    iterations = parse(Int, ARGS[2])
    length_arg = ispow2(length_arg) ? length_arg : round_to_power2(length_arg)
    length_arg = max(length_arg, 256)
    tolerance = Float32(0.001)

    rng = MersenneTwister(123)
    diagonal = Float32.(floor.(rand(rng, length_arg) .* 256))
    rng = MersenneTwister(133)
    off_diagonal = Float32.(floor.(rand(rng, length_arg - 1) .* 256))

    lower, upper = compute_gerschgorin(diagonal, off_diagonal, length_arg)
    eigen_intervals = [fill(upper, 2 * length_arg), fill(upper, 2 * length_arg)]
    eigen_intervals[1][1] = lower
    eigen_intervals[1][2] = upper

    d_diagonal = CuArray(diagonal)
    d_off_diagonal = CuArray(off_diagonal)
    d_num_intervals = CUDA.zeros(UInt32, length_arg)
    d_buffers = [CUDA.zeros(Float32, 2 * length_arg), CUDA.zeros(Float32, 2 * length_arg)]

    for _ in 1:2
        run_kernels!(d_diagonal, d_num_intervals, d_off_diagonal, d_buffers,
                     eigen_intervals, length_arg, tolerance)
    end

    println("Executing kernel for $iterations iterations")
    println("-------------------------------------------")
    CUDA.synchronize()
    start = time_ns()
    in_idx = 1
    for _ in 1:iterations
        in_idx = run_kernels!(d_diagonal, d_num_intervals, d_off_diagonal, d_buffers,
                              eigen_intervals, length_arg, tolerance)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / iterations
    @printf("Average kernel execution time %f (us)\n", elapsed_us)

    verification = [fill(upper, 2 * length_arg), fill(upper, 2 * length_arg)]
    verification[1][1] = lower
    verification[1][2] = upper
    verification_in = 1
    while is_complete(verification[verification_in], length_arg, tolerance)
        eigenvalue_cpu_reference!(diagonal, off_diagonal, length_arg,
                                  verification[verification_in],
                                  verification[3 - verification_in],
                                  tolerance)
        verification_in = 3 - verification_in
    end

    println(compare_intervals(eigen_intervals[in_idx], verification[verification_in]) ? "PASS\n" : "FAIL\n")
end

main()
