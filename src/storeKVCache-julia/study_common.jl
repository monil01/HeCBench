using CUDA
using Printf
function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");batch=256;cache=512;widths=(16,32,64,128);fixtures=[]
    for row in widths
        loc=UInt64[UInt64((i*73+19)%cache) for i in 0:batch-1];k=UInt64[UInt64(i)*UInt64(0x9e3779b97f4a7c15)+UInt64(17) for i in 0:batch*row-1];v=UInt64[~x for x in k];ek=zeros(UInt64,cache*row);ev=similar(ek);fill!(ev,0)
        for i in 1:batch,j in 1:row
            ek[Int(loc[i])*row+j]=k[(i-1)*row+j];ev[Int(loc[i])*row+j]=v[(i-1)*row+j]
        end
        kc,vc=CUDA.zeros(UInt64,cache*row),CUDA.zeros(UInt64,cache*row);dl,dk,dv=CuArray(loc),CuArray(k),CuArray(v);launch! = prepare(kc,vc,dl,dk,dv,batch,row);study_loop!(launch!,1);push!(fixtures,(kc,vc,launch!,vcat(ek,ev)))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=UInt64[];times=Float64[];ok=true;gct=0.0;gcp=0
        for (kc,vc,launch!,expected) in fixtures
            CUDA.fill!(kc,UInt64(0));CUDA.fill!(vc,UInt64(0));CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;got=vcat(Array(kc),Array(vc));ok &=got==expected;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"storeKVCache_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_four_dispatch_width_kernel_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"width_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),repeat,times[1],times[2],times[3],times[4],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
