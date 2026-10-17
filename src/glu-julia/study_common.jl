using CUDA
using Printf
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");dim=32;input=Float32[Float32((i*17+3)%1025)/1024f0*12f0-6f0 for i in 0:dim^3-1];di=CuArray(input);fixtures=[]
    for axis in (2,0,1,2,0,1,2)
        m=dim^axis;s=dim÷2;n=dim^(2-axis);expected=zeros(Float32,dim^3÷2)
        for idx in 0:length(expected)-1
            outer=idx÷(s*n);mid=(idx÷n)%s;inner=idx%n;x1=input[outer*2*s*n+mid*n+inner+1];x2=input[outer*2*s*n+(mid+s)*n+inner+1];expected[idx+1]=Float32(Float64(x1)/(1.0+exp(-Float64(x2))))
        end
        dout=CUDA.zeros(Float32,length(expected));launch! = prepare(di,dout,m,s,n);launch!();push!(fixtures,(dout,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0
        for (dout,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dout);err=maximum(abs.(got.-expected));ok &=all(isfinite,got)&&err<=1e-3;ma=max(ma,err);append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"glu_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_seven_original_axis_loop_GLU_operations_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
