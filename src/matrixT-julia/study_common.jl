using CUDA
using Printf
function matrix_gold(input,N,kind)
    out=zeros(Float32,N*N)
    for y in 0:N-1,x in 0:N-1
        if kind<=1
            out[y*N+x+1]=input[y*N+x+1]
        elseif kind==2
            ox=(y÷16)*16+x%16;oy=(x÷16)*16+y%16;out[oy*N+ox+1]=input[y*N+x+1]
        elseif kind==3
            sx=(x÷16)*16+y%16;sy=(y÷16)*16+x%16;out[y*N+x+1]=input[sy*N+sx+1]
        else
            out[x*N+y+1]=input[y*N+x+1]
        end
    end
    out
end
function study_main(prepare,args=ARGS)
    N=128;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");input=Float32.(0:N*N-1);dx=CuArray(input);fixtures=[]
    for kind in 0:8
        expected=matrix_gold(input,N,kind);out=CUDA.fill(Float32(NaN),N*N);launch! = prepare(out,dx,N,kind);launch!();push!(fixtures,(out,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true
        for (out,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(out);ok &=got==expected;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"matrixT_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_all9_copy_partial_and_complete_transpose_kernels\",\"jit_excluded\":true,\"internal_repeats\":%d,\"case_times_seconds\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),repeat,string(times),file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
