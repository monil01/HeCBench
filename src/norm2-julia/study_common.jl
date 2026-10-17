using CUDA
using Printf
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for n in (524288,1048576)
        input=Float32[Float32((i+1)%7) for i in 0:n-1];gold=Float32(sqrt(sum(Float64(x)*Float64(x) for x in input)));da=CuArray(input);getnorm! = prepare(da,n);getnorm!();push!(fixtures,(getnorm!,gold))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0
        for (getnorm!,gold) in fixtures
            got=zeros(Float32,repeat);CUDA.synchronize();t0=time_ns()
            for rep in 1:repeat
                got[rep]=getnorm!()
            end
            CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);err=maximum(abs.(got.-gold));ma=max(ma,err);ok &=all(isfinite,got)&&err<=.1;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"norm2_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_two_sizes_full_norm_plus_host_scalar_result_every_call\",\"jit_excluded\":true,\"internal_repeats\":%d,\"size_times_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,times[1],times[2],file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
