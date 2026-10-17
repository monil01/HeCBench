using CUDA
using Printf
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for heads in (6,12,24,48)
        sz0=2*heads;sz2=128;seq=32;ccall(:srand,Cvoid,(Cuint,),Cuint(heads));s1=ccall(:rand,Cint,())%(seq-1)+1;s2=seq-s1;n1=sz0*sz2*s1;n2=sz0*sz2*s2
        a=Float32[Float32(ccall(:rand,Cint,())%n1) for _ in 1:n1];b=Float32[Float32(ccall(:rand,Cint,())%n2) for _ in 1:n2];expected=zeros(Float32,n1+n2)
        for outer in 0:sz0-1,mid in 0:seq-1,inner in 0:sz2-1
            expected[(outer*seq+mid)*sz2+inner+1]=mid<s1 ? a[(outer*s1+mid)*sz2+inner+1] : b[(outer*s2+mid-s1)*sz2+inner+1]
        end
        da,db,dc=CuArray(a),CuArray(b),CUDA.zeros(Float32,length(expected));launch! = prepare(da,db,dc,sz0,sz2,s1,s2);launch!();push!(fixtures,(dc,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0
        for (dc,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dc);err=maximum(abs.(got.-expected));ok &=all(isfinite,got)&&err==0;ma=max(ma,err);append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"concat_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_four_head_count_concat_operations_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"head_count_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,times[1],times[2],times[3],times[4],file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
