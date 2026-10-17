using CUDA
using Printf
function study_loop!(launch!,mode,repeat)
    for _ in 1:repeat
        launch!(mode)
    end
end
function study_main(prepare,args=ARGS)
    n=262144;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");margin=.01f0
    y=Int32[isodd(i) ? -1 : 1 for i in 0:n-1];x1=Float32[Float32((13i+7)%257-128)/64f0 for i in 0:n-1];x2=Float32[Float32((37i+11)%257-128)/64f0 for i in 0:n-1];dout=Float32[Float32((29i+3)%257-128)/64f0 for i in 0:n-1]
    # Include exact hinge equality and both branch signs with nonzero output gradients.
    for i in 1:256
        y[i]=1;x1[i]=margin;x2[i]=0f0;dout[i]=.5f0
    end
    r1=zeros(Float32,n);r2=similar(r1)
    for i in 1:n
        active=-Float32(y[i])*(x1[i]-x2[i])+margin>=0f0;r1[i]=active ? -Float32(y[i])*dout[i] : 0f0;r2[i]=active ? Float32(y[i])*dout[i] : 0f0
    end
    expected=vcat(r1,r2);dy,da,db,dd=CuArray(y),CuArray(x1),CuArray(x2),CuArray(dout);o1,o2=CUDA.zeros(Float32,n),CUDA.zeros(Float32,n);launch! = prepare(dy,da,db,dd,o1,o2,n,margin)
    for mode in 1:3
        study_loop!(launch!,mode,1)
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ma=0f0;mr=0f0;ok=true;gct=0.0;gcp=0
        for mode in 1:3
            CUDA.fill!(o1,NaN32);CUDA.fill!(o2,NaN32);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,mode,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;got=vcat(Array(o1),Array(o2));err=abs.(got.-expected);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(expected),eps(Float32))));ok &=all(isfinite,got)&&maximum(err)<=1f-3;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"mrc_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_three_MRC_kernel_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"mode_times_seconds\":[%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,mr,repeat,times[1],times[2],times[3],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
