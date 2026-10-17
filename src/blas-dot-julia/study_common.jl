using CUDA
using Printf
function study_main(prepare,args=ARGS)
    n=256;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for T in (Float64,Float32,Float16,CUDA.CUDACore.BFloat16)
        x=T(sqrt(65504.0/n));a=fill(x,n);b=copy(a);expected=sum(Float64(Float32(v)*Float32(v)) for v in a)
        da,db,dr=CuArray(a),CuArray(b),CUDA.zeros(T,1);launch! = prepare(da,db,dr,n);launch!();push!(fixtures,(dr,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float64[];times=Float64[];ma=0.0;ok=true
        for (dr,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns()
            for rep in 1:repeat
                launch!()
            end
            CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Float64(only(Array(dr)));err=abs(got-expected);ma=max(ma,err);ok &=isfinite(got)&&err<0.1;push!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"blas_dot_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":4,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_FP64_FP32_FP16_BF16_dot_device_result_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"dtype_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":4,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),ma,repeat,times[1],times[2],times[3],times[4],file)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
