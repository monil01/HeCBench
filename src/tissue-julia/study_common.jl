using CUDA
using Printf
function study_main(prepare,args=ARGS)
    dim=8;nnt=dim^3;capacity=32768;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");points=zeros(Int32,3capacity);prev=zeros(Float32,capacity);qt=zeros(Float32,capacity)
    for i in 0:nnt-1
        points[i+1]=i%dim;points[nnt+i+1]=(i÷dim)%dim;points[2nnt+i+1]=i÷(dim*dim);prev[i+1]=Float32(i%9)/8f0;qt[i+1]=Float32(i%13-6)/16f0
    end
    g=Float32[Float32(i%17+1)/32f0 for i in 0:2capacity-1];b=Float32[Float32(i%11+1)/64f0 for i in 0:2capacity-1];dp,dg,db,dprev,dqt=CuArray(points),CuArray(g),CuArray(b),CuArray(prev),CuArray(qt);fixtures=[]
    for species in 1:2
        gold=zeros(Float64,capacity)
        for i in 0:nnt-1
            s=0.0
            for j in 0:nnt-1
                dist=abs(points[j+1]-points[i+1])+abs(points[nnt+j+1]-points[nnt+i+1])+abs(points[2nnt+j+1]-points[2nnt+i+1])+(species-1)*capacity;idx=Int(dist)+1;s+=Float64(g[idx])*Float64(prev[j+1])+Float64(b[idx])*Float64(qt[j+1])
            end
            gold[i+1]=s
        end
        dc=CUDA.zeros(Float32,capacity);launch! = prepare(dp,dg,db,dc,dprev,dqt,nnt,capacity,species);launch!();push!(fixtures,(dc,launch!,gold))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0
        for (dc,launch!,gold) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dc);err=maximum(abs.(Float64.(got).-gold));ma=max(ma,err);ok &=all(isfinite,got)&&err==0&&all(iszero,view(got,nnt+1:capacity));append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"tissue_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_two_species_complete_nonlocal_tissue_solute_operator_native_step1\",\"jit_excluded\":true,\"internal_repeats\":%d,\"written_output_values\":1024,\"unchanged_zero_tail_values\":64512,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
