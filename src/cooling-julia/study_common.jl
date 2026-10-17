using CUDA
using Printf
include(joinpath(@__DIR__,"cooling_math.jl"))
function study_main(prepare,args=ARGS)
    n=512;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");temperature=[10.0^(3.0+5.0*i/(n-1)) for i in 0:n-1];dt=CuArray(temperature);fixtures=[]
    for heat in Int32[0,1]
        dr=CUDA.zeros(Float64,n);expected=[primordial_cool(.0899,t,heat) for t in temperature];launch! = prepare(dt,dr,n,heat);launch!();push!(fixtures,(dr,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float64[];seconds=0.0;ma=0.0;mr=0.0;ok=true
        for (dr,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/repeat
            got=Array(dr);err=abs.(got.-expected);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(expected),1e-300)));ok &=all(isfinite,got)&&all(isfinite,expected)&&all(err.<=1e-34.+1e-10.*abs.(expected));append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"cooling_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":1024,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_full_primordial_cooling_heat0_heat1_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":1024,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),ma,mr,repeat,file)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
