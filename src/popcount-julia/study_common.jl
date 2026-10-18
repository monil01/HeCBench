using CUDA,Printf
include(joinpath(@__DIR__,"study_math.jl"))
function popcount_study(factory)
    n=4099;repeats=11;guard=Int32(-1234567890)
    host=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"study_input.bin"))))
    length(host)==n || error("fixture size")
    # Independent fixed64-bit expansion; no SWAR/clear-lowest-bit/intrinsic oracle.
    ref=Int32[sum(Int32((x >> k)&1) for k in 0:63) for x in host]
    table=UInt8[sum(Int((x>>k)&1) for k in 0:7) for x in 0:255]
    data=CuArray(host);out=CUDA.fill(guard,6*(n+2));lut=CUDA.zeros(UInt8,256)
    launch! = factory(data,out,lut,n)
    function suite!()
        for which in 1:6
            for _ in 1:repeats
                if which==5;copyto!(lut,table);end
                launch!(Int32(which))
            end
        end
        CUDA.device_synchronize(blocking=true)
    end
    suite!()
    allok=true
    for trial in 1:5
        fill!(out,guard);CUDA.device_synchronize(blocking=true)
        t=time_ns();suite!();secs=(time_ns()-t)*1e-9/repeats
        got=Array(out);actual_input=Array(data)
        ok=actual_input==host
        for which in 0:5
            start=which*(n+2)
            ok &= got[start+1]==guard && got[start+n+2]==guard
            ok &= got[start+2:start+n+1]==ref
        end
        # Exact input bits followed by every output and both per-algorithm guards.
        artifact=vcat(collect(reinterpret(Int32,actual_input)),got)
        path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
        allok &=ok
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"six_algorithms_including_pc5_LUT_upload_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(artifact),ok ? "0" : "null",repeats,path,length(artifact))
    end
    allok||error("complete bitcount/input/guard check failed");println("PASS")
end
