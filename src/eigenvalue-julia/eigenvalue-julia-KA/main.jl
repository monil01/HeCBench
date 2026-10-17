using CUDA,KernelAbstractions



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

@kernel function cal_num_eigenvalue_interval_kernel!(num_intervals, eigen_intervals,
                                             diagonal, off_diagonal,
                                             width::UInt32)
    gid0 = UInt32(@index(Global,Linear))-UInt32(1)
    lower_id = UInt32(2) * gid0
    upper_id = lower_id + UInt32(1)
    lower_limit = eigen_intervals[Int(lower_id) + 1]
    upper_limit = eigen_intervals[Int(upper_id) + 1]
    lower = cal_num_less_kernel(lower_limit, width, diagonal, off_diagonal)
    upper = cal_num_less_kernel(upper_limit, width, diagonal, off_diagonal)
    num_intervals[Int(gid0) + 1] = upper - lower
end

@kernel function recalculate_eigen_intervals_kernel!(new_intervals, eigen_intervals,
                                             num_intervals, diagonal,
                                             off_diagonal, width::UInt32,
                                             tolerance::Float32)
    gid0 = UInt32(@index(Global,Linear))-UInt32(1)
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
end

include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(diagonal,off,counts,buffers)
 k1=cal_num_eigenvalue_interval_kernel!(get_backend(diagonal),256);k2=recalculate_eigen_intervals_kernel!(get_backend(diagonal),256)
 return idx->begin
 k1(counts,buffers[idx],diagonal,off,UInt32(256);ndrange=256)
 k2(buffers[3-idx],buffers[idx],counts,diagonal,off,UInt32(256),0.001f0;ndrange=256)
 end
end
eigenvalue_study(factory)
