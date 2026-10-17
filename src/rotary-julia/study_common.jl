using CUDA
using Printf
function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    n=262144;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat")
    x1=Float32[Float32((i*13+7)%257-128)/128f0 for i in 0:n-1];x2=Float32[Float32((i*37+11)%257-128)/128f0 for i in 0:n-1];cosv=Vector{Float32}(undef,n);sinv=similar(cosv)
    for i in 0:n-1
        angle=Float32(i÷64)/ccall((:powf,"libm.so.6"),Cfloat,(Cfloat,Cfloat),10000f0,2f0*Float32(i%64)/128f0)
        cosv[i+1]=ccall((:cosf,"libm.so.6"),Cfloat,(Cfloat,),angle);sinv[i+1]=ccall((:sinf,"libm.so.6"),Cfloat,(Cfloat,),angle)
    end
    expected=vcat(x1.*cosv.-x2.*sinv,x1.*sinv.+x2.*cosv);o1,o2=CUDA.zeros(Float32,n),CUDA.zeros(Float32,n);a,b,c,s=CuArray(x1),CuArray(x2),CuArray(cosv),CuArray(sinv);launch! = prepare(o1,o2,a,b,c,s,n);study_loop!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num()
        got=vcat(Array(o1),Array(o2));err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));ok=all(isfinite,got)&&ma<=1f-3;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"rotary_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"rotary_kernel_only_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,mr,repeat,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
