module OwnedStudy
using CUDA, Printf
"""Five in-process synchronized host-clock trials, JIT warmed before trial 1.
The same timed launch loop and reset/output boundaries apply to all models.
Full-array CPU checks are outside timing; raw outputs enable dual comparisons.
"""
function run_trials!(launch!, reset!, readback, expected, repeats; atol=0.0, rtol=0.0)
    repeats > 0 || error("positive internal repeat count required")
    reset!(); launch!(); CUDA.synchronize()
    allok = true
    for trial in 1:5
        reset!(); CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeats
            launch!()
        end
        CUDA.synchronize()
        secs = (time_ns()-start)*1e-9/repeats
        got = vec(readback()); ref = vec(expected)
        length(got)==length(ref) || error("output shape mismatch")
        maxabs=0.0; maxrel=0.0; ok=true
        for (x,y) in zip(got,ref)
            a=abs(Float64(x)-Float64(y))
            b=a/max(abs(Float64(y)),atol,1e-300)
            ok &= isfinite(x) && isfinite(y) && a<=atol+rtol*abs(y)
            maxabs=max(maxabs,a); maxrel=max(maxrel,b)
        end
        allok &= ok
        name="output_trial$(trial).bin"
        path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),name)
        write(path,got)
        dtype=eltype(got)==Float32 ? "float32" : eltype(got)==Float64 ? "float64" : eltype(got)==Int32 ? "int32" : eltype(got)==Int64 ? "int64" : error("unsupported dtype")
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"%s\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(got),maxabs,maxrel,repeats,path,dtype,length(got))
    end
    allok || error("CPU numerical verification failed")
    println("PASS")
end
end
