using CUDA
import JACC
JACC.@init_backend
using Printf
using Random
# Native libstdc++ default_random_engine=minstd_rand0, uniform_int_distribution.
function native_counts(n)
    state = Int64(123)
    scaling = div(Int64(2147483645), 9900)
    past = scaling * 9900
    counts = Int[]
    while length(counts) < n
        state = mod(16807 * state, 2147483647)
        value = state - 1
        value < past && push!(counts, 100 + div(value, scaling))
    end
    counts
end

function reverse_kernel!(i, d, len::Int32)
    shared = JACC.shared(d)
    @inbounds d[i] = shared[len-i+1]
    return
end

function run_reverse(iteration::Int)
    len = 256
    gold_even = Int32.(collect(0:len-1))
    gold_odd = Int32.(collect(len-1:-1:0))
    test = Vector{Int32}(undef, len)
    d_test = CuArray(gold_even)

    counts = native_counts(iteration)
    spec=JACC.launch_spec(threads=256,blocks=1,shmem_size=1024,sync=false)
    # Device compilation and launch initialization precede all measured trials.
    JACC.parallel_for(spec, len, reverse_kernel!, d_test, Int32(len))
    CUDA.synchronize()
    for trial in 1:5
      total_ns = 0
      error = false
      for count in counts
        copyto!(d_test, gold_even)

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:count
            JACC.parallel_for(spec, len, reverse_kernel!, d_test, Int32(len))
        end
        CUDA.synchronize()
        total_ns += time_ns() - start

        copyto!(test, d_test)
        expected = iseven(count) ? gold_even : gold_odd
        if test != expected
            error = true
            break
        end
    end

      artifact = joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()), "trial_$(trial).bin")
      open(artifact, "w") do io
          write(io, test)
      end
      @printf("Total kernel execution time: %f (s)\n", total_ns * 1e-9)
      println(error ? "FAIL" : "PASS")
      println("STUDY_TRIAL {\"trial\":", trial, ",\"time_seconds\":", total_ns*1e-9,
              ",\"correct\":", !error, ",\"checked_values\":", len*iteration,
              ",\"max_abs_error\":", error ? "null" : "0.0",
              ",\"max_rel_error\":", error ? "null" : "0.0",
              ",\"timing_region\":\"synchronized_kernel_launch_loops_total\",\"jit_excluded\":true,\"internal_repeats\":", sum(counts), ",\"output_artifact\":{\"path\":\"", artifact, "\",\"dtype\":\"int32\",\"count\":256,\"byte_order\":\"little\"}}")
      error && return 1
    end
    return 0
end

function main()
    if length(ARGS) != 1
        println("Usage: ./main.jl <iterations>")
        return 1
    end

    iteration = parse(Int, ARGS[1])
    return run_reverse(iteration)
end

exit(main())
