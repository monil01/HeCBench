using CUDA
using KernelAbstractions
using Printf

const RADIUS = 7
const BLOCK_SIZE = 256

# Native CUDA uses a cooperative shared-memory tile. Preserve that mapping.
@kernel function stencil_1d!(input, output, n)
    g = @index(Global, Linear)
    l = @index(Local, Linear)
    tile = @localmem Int32 (270,)
    @inbounds tile[l + RADIUS] = input[g]
    if l <= RADIUS
        @inbounds tile[l] = g <= RADIUS ? Int32(0) : input[g - RADIUS]
        @inbounds tile[l + BLOCK_SIZE + RADIUS] = input[g + BLOCK_SIZE]
    end
    @synchronize
    result = Int32(0)
    for offset in -RADIUS:RADIUS
        @inbounds result += tile[l + RADIUS + offset]
    end
    @inbounds output[g] = result
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
    kernel = stencil_1d!(CUDABackend(), BLOCK_SIZE)
    kernel(din, dout, n; ndrange=n)
    CUDA.synchronize() # Explicit device JIT warmup outside all measured trials.
    all_ok = true
    for trial in 1:5
        CUDA.fill!(dout, Int32(0)); CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:repeats
            kernel(din, dout, n; ndrange=n)
        end
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
