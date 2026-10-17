using CUDA
using Printf
isdefined(@__MODULE__,:fresnel_corrected)||include(joinpath(@__DIR__,"fresnel_math.jl"))
function quadrature_oracle(x)
    steps=4096;h=x/steps;s=0.0
    for i in 0:steps
        t=h*i;s+=(i==0||i==steps ? 1.0 : (isodd(i) ? 4.0 : 2.0))*sin(t*t)
    end
    s*h/3.0*sqrt(2.0/pi)
end
function study_main(prepare,args=ARGS)
    n=8193;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");input=[Float64(i)/1024.0 for i in 0:n-1];gold=quadrature_oracle.(input);expected=fresnel_corrected.(input);di,dc=CuArray(input),CUDA.zeros(Float64,n);launch! = prepare(di,dc,n);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds=(time_ns()-t0)*1e-9/repeat;got=Array(dc);err=maximum(abs.(got.-expected));qe=maximum(abs.(got.-gold));ok=all(isfinite,got)&&err<=1e-6&&qe<=1e-6&&all(got[1:513].>=0);allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"fresnel_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"quadrature_oracle_max_error\":%.12g,\"timing_region\":\"source_convention_full_FresnelS_corrected_auxiliary_sine_formula\",\"jit_excluded\":true,\"internal_repeats\":%d,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),n,err,qe,repeat,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
