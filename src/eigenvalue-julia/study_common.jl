using CUDA
include(joinpath(@__DIR__,"study_eigenvalue_oracle.jl"))
const STUDY_REGION="restarted_complete_host_transfer_sturm_bisection_solve_per_repeat"
include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"),String),"kernel_launch_loop_per_internal_repeat"=>STUDY_REGION),joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
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

function eigenvalue_study(factory)
 n=256;repeats=10;tol=0.001f0
 diagonal=Float32[4f0+Float32(i)/2f0 for i in 0:n-1];off=Float32[Float32(8+mod(i,5))/256f0 for i in 0:n-2]
 lower,upper=compute_gerschgorin(diagonal,off,n);initial=fill(upper,2*n);initial[1]=lower
 cpu=[copy(initial),copy(initial)];ci=1;cpuiters=0
 while is_complete(cpu[ci],n,tol)
  eigenvalue_cpu_reference!(diagonal,off,n,cpu[ci],cpu[3-ci],tol);ci=3-ci;cpuiters+=1;cpuiters<100||error("CPUconvergence")
 end
 expected=vcat(cpu[ci],ones(Float32,n));dd=CuArray(diagonal);doff=CuArray(off);counts=CUDA.zeros(UInt32,n);buffers=[CUDA.zeros(Float32,2*n),CUDA.zeros(Float32,2*n)];host=[copy(initial),copy(initial)];lastindex=Ref(1);call=Ref(0);history=zeros(Int,51)
 pair! = factory(dd,doff,counts,buffers)
 launch! = ()->begin
  copyto!(host[1],initial);copyto!(host[2],initial)
  copyto!(buffers[1],host[1]);copyto!(buffers[2],host[2]);idx=1;steps=0
  while is_complete(host[idx],n,tol)
   pair!(idx);idx=3-idx;copyto!(host[idx],buffers[idx]);steps+=1;steps<100||error("GPUconvergence")
  end
  lastindex[]=idx;call[]+=1;history[call[]]=steps
 end
 readback=()->begin
  intervals=host[lastindex[]];distance=0.;maxwidth=0.
  for i in 1:n
   lo=Float64(intervals[2*i-1]);hi=Float64(intervals[2*i]);ev=INDEPENDENT_EIGENVALUES[i]
   isfinite(lo)&&isfinite(hi)&&lo<=hi&&hi-lo<Float64(tol)||error("intervalwidth/finite")
   distance=max(distance,max(lo-ev,ev-hi,0.));maxwidth=max(maxwidth,hi-lo)
  end
  distance<=1e-4||error("independentspectralenclosure")
  compare_intervals(cpu[ci],intervals)||error("originalrelativeL2criterion")
  println("STUDY_EIGEN_ACCURACY max_enclosure_distance=",distance," max_width=",maxwidth," CPU_iterations=",cpuiters," last10_solve_iterations=",history[call[]-9:call[]])
  vcat(copy(intervals),Float32.(Array(counts)))
 end
 OwnedStudy.run_trials!(launch!,()->nothing,readback,expected,repeats;atol=1e-6,rtol=0.)
end
