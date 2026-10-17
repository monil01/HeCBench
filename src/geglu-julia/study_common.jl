using CUDA
using SpecialFunctions
using Printf
function study_main(prepare,args=ARGS)
    rows=128;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for width in (1280,2560,5120)
        input=Float32[Float32((i*17+3)%1025)/1024f0*12f0-6f0 for i in 0:2*rows*width-1];expected=zeros(Float32,rows*width)
        for row in 0:rows-1,col in 0:width-1
            x=Float64(input[row*2*width+col+1]);g=Float64(input[row*2*width+width+col+1]);expected[row*width+col+1]=Float32(x*g*.5*(1+erf(g/sqrt(2.))))
        end
        di,dc=CuArray(input),CUDA.zeros(Float32,length(expected));launch! = prepare(di,dc,rows,width);launch!();push!(fixtures,(dc,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ma=0.0;ok=true
        for (dc,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dc);err=maximum(abs.(got.-expected));ma=max(ma,err);ok &=all(isfinite,got)&&err<=1e-3;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"geglu_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_three_supported_widths_full_erf_GeGLU_operator\",\"jit_excluded\":true,\"internal_repeats\":%d,\"width_times_seconds\":[%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,times[1],times[2],times[3],file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
