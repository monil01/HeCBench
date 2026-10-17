using CUDA
using Printf
function study_loop!(launch!,method,repeat)
    for _ in 1:repeat
        launch!(method)
    end
end
function study_main(prepare,args=ARGS)
    nr=257;nc=263;bc=256;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");rp=Int32[0];ci=Int32[]
    for row in 0:nr-1
        count=row%65
        for j in 0:count-1
            push!(ci,Int32((row*17+j*19)%nc))
        end
        push!(rp,Int32(length(ci)))
    end
    av=ones(Float32,length(ci));dense=Float32[Float32((i*37+17)%129-64)/64f0 for i in 0:nc*bc-1];expected=zeros(Float32,nr*bc)
    for row in 0:nr-1,col in 0:bc-1
        acc=0f0
        for k in Int(rp[row+1])+1:Int(rp[row+2])
            acc+=av[k]*dense[Int(ci[k])*bc+col+1]
        end
        expected[row*bc+col+1]=acc
    end
    dr,dc,da,db=CuArray(rp),CuArray(ci),CuArray(av),CuArray(dense);out=CUDA.zeros(Float32,nr*bc);launch! = prepare(dr,dc,da,db,out,nr,bc)
    for method in 1:4
        study_loop!(launch!,method,1)
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ma=0f0;mr=0f0;ok=true;gct=0.0;gcp=0
        for method in 1:4
            CUDA.fill!(out,NaN32);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,method,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;got=Array(out);err=abs.(got.-expected);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(expected),eps(Float32))));ok &=all(isfinite,got)&&maximum(err)<=1f-2;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"ge_spmm_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_four_CSR_SpMM_methods_kernel_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"method_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,mr,repeat,times[1],times[2],times[3],times[4],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
