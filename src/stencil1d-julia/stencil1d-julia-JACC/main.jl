using CUDA
using JACC
using Printf

const RADIUS = 7
const BLOCK_SIZE = 256

# High-level JACC map retains the CUDA.jl global-memory implementation.
# Native CUDA/KA cooperative tile is a documented mapping difference.
function stencil_1d!(i, input, output)
    result = Int32(0)
    for offset in -RADIUS:RADIUS
        j = i + offset
        if j > 0
            @inbounds result += input[j]
        end
    end
    @inbounds output[i] = result
    return nothing
end

function launch_batch!(spec, din, dout, n, repeats)
    for _ in 1:repeats
        JACC.parallel_for(stencil_1d!, spec, n, din, dout)
    end
    return nothing
end

function reference(input, n)
    out = zeros(Int32, n)
    for i in 1:n, off in -RADIUS:RADIUS
        j = i + off
        j > 0 && (out[i] += input[j])
    end
    out
end

function main(args=ARGS)
    n = length(args) >= 1 ? parse(Int, args[1]) : 1048576
    repeats = length(args) >= 2 ? parse(Int, args[2]) : 100
    n > 0 && n % BLOCK_SIZE == 0 && repeats > 0 || error("positive length multiple of 256 and positive repeat required")
    input = Int32.(0:n+RADIUS-1)
    expected = reference(input, n)
    din = CuArray(input); dout = CUDA.zeros(Int32, n)
    backend = JACC.get_backend(:cuda)
    spec = JACC.LaunchSpec{typeof(backend)}(; threads=BLOCK_SIZE, blocks=cld(n,BLOCK_SIZE), shmem_size=0, sync=false)
    launch_batch!(spec, din, dout, n, 1)
    CUDA.synchronize() # Warm the exact host launch path and device kernel.
    all_ok = true
    for trial in 1:5
        CUDA.fill!(dout, Int32(0)); CUDA.synchronize()
        t0 = time_ns()
        launch_batch!(spec, din, dout, n, repeats)
        CUDA.synchronize()
        elapsed = (time_ns()-t0)*1e-9/repeats
        result = Array(dout)
        maxabs = maximum(abs.(Int64.(result).-Int64.(expected)))
        ok = result == expected
        all_ok &= ok
        artifact = joinpath(get(ENV, "STUDY_OUTPUT_DIR", pwd()), "stencil1d_trial$(trial).bin"); write(artifact, result)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0.0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n", trial, elapsed, string(ok), n, maxabs, repeats, artifact, n)
    end
    println(all_ok ? "PASS" : "FAIL")
    return all_ok ? 0 : 1
end

exit(main())
