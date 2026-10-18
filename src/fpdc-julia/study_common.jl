using CUDA,Printf
const FPDCROOT=@__DIR__
function fpdc_study(factory)
    inputbase=CUDA.fill(UInt64(0xa55aa55aa55aa55a),544);bytebase=CUDA.fill(UInt8(165),4416);cutbase=CUDA.fill(Int32(0x13572468),40);offbase=CUDA.fill(Int32(0x13572468),40)
    input=view(inputbase,17:528);bytes=view(bytebase,33:4384);cuts=view(cutbase,17:24);offsets=view(offbase,17:24)
    hostcuts=collect(reinterpret(Int32,read(joinpath(FPDCROOT,"study_cuts_i32.bin"))));copyto!(cuts,hostcuts)
    fixtures=map(i->collect(reinterpret(UInt64,read(joinpath(FPDCROOT,"study_case$(i)_u64.bin")))),(1,2))
    cases=[(c,d,read(joinpath(FPDCROOT,"oracle_case$(c)_d$(d)_payload.bin")),collect(reinterpret(Int32,read(joinpath(FPDCROOT,"oracle_case$(c)_d$(d)_off_i32.bin"))))) for c in (1,2) for d in (1,2,3,32)]
    run=factory(input,bytes,cuts,offsets);copyto!(input,fixtures[1])
    for d in (1,2,3,32);run(Int32(d));CUDA.device_synchronize(blocking=true);end
    allok=true
    for trial in 1:5
        prefix=UInt64[];outputs=UInt64[];guards=UInt64[];times=Float64[];ok=true;mismatch=0
        for (c,d,expected,eo) in cases
            copyto!(input,fixtures[c]);fill!(bytes,UInt8(165));fill!(offsets,Int32(0x13572468));CUDA.device_synchronize(blocking=true)
            start=time_ns()
            for _ in 1:3;run(Int32(d));CUDA.device_synchronize(blocking=true);end
            push!(times,(time_ns()-start)*1e-9/3)
            ai=Array(input);ac=Array(cuts);actual=Array(bytes);off=Array(offsets);ok&=ai==fixtures[c]&&ac==hostcuts
            append!(prefix,ai);append!(prefix,UInt64.(ac));append!(prefix,UInt64[512,4,2,d,3,64,4352,8])
            append!(outputs,actual);append!(outputs,UInt64.(reinterpret(UInt32,off)));ok&=actual==expected&&off==eo;mismatch+=sum(actual.!=expected)+sum(off.!=eo)
            for base in (inputbase,bytebase,cutbase,offbase)
                raw=Array(base);g=base===bytebase ? 32 : 16;sent=base===inputbase ? UInt64(0xa55aa55aa55aa55a) : (base===bytebase ? UInt64(165) : UInt64(0x13572468))
                guard=vcat(raw[1:g],raw[end-g+1:end]);ok&=all(x->UInt64(x)==sent,guard);append!(guards,UInt64.(guard))
            end
        end
        allok&=ok;artifact=vcat(prefix,outputs,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":36160,\"max_abs_error\":%s,\"max_rel_error\":null,\"mismatch_values\":%d,\"timing_region\":\"sum8_fpdc_compression_kernel_launch_loop_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_cases\":8,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ok ? "0" : "null",mismatch,path,length(artifact))
    end
    allok||error("Full codec payload/offset/input/guard mismatch");println("PASS")
end
