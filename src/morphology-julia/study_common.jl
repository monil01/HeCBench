using CUDA
using Printf
const STUDY_W=510
const STUDY_H=510
const STUDY_SEL=5
function study_axis_cpu!(dst,src,W,H,sel,op,axis)
    half=sel÷2
    @inbounds for y in 0:H-1,x in 0:W-1
        i=y*W+x+1;p=axis==1 ? x : y;limit=axis==1 ? W : H
        if p<half||p>=limit-half
            dst[i]=axis==1 ? UInt8(0) : (op==1 ? UInt8(255) : UInt8(0))
        else
            acc=op==1 ? UInt8(0) : UInt8(255)
            for k in -half:half
                j=axis==1 ? i+k : i+k*W;v=src[j];acc=op==1 ? max(acc,v) : min(acc,v)
            end
            dst[i]=acc
        end
    end
end
function fixture_images()
    W=STUDY_W;H=STUDY_H;n=W*H;a=zeros(UInt8,n);a[(H÷2-1)*W+(W÷2-1)+1]=255
    b=UInt8[UInt8(mod(i*37+(i÷W)*13+17,256)) for i in 0:n-1];[a,b]
end
function run_cycles!(cycle!,img,tmp,repeats)
    for _ in 1:repeats
        cycle!(img,tmp)
    end
    nothing
end
function study_main(prepare,args=ARGS)
    repeats=length(args)>0 ? parse(Int,args[1]) : 10;repeats>0||error("positive repeatsrequired")
    inputs=fixture_images();expected=Vector{Vector{UInt8}}()
    for input in inputs
        a=copy(input);b=similar(a)
        for _ in 1:repeats,op in (1,2)
            study_axis_cpu!(b,a,STUDY_W,STUDY_H,STUDY_SEL,op,1);study_axis_cpu!(a,b,STUDY_W,STUDY_H,STUDY_SEL,op,2)
        end
        push!(expected,a)
    end
    a=CuArray(inputs[1]);b=similar(a);cycle! = prepare(a,b)
    run_cycles!(cycle!,a,b,1);CUDA.synchronize();allok=true
    for trial in 1:5
        allgot=UInt8[];total=0.0;maxabs=0;gcsecs=0.0;gcpauses=0;ok=true
        for (case,input) in enumerate(inputs)
            copyto!(a,input);CUDA.fill!(b,UInt8(0));CUDA.synchronize();gb=Base.gc_num()
            t0=time_ns();run_cycles!(cycle!,a,b,repeats);CUDA.synchronize();total+=(time_ns()-t0)*1e-9
            ga=Base.gc_num();gcsecs+=(ga.total_time-gb.total_time)*1e-9;gcpauses+=ga.pause-gb.pause
            got=Array(a);maxabs=max(maxabs,maximum(abs.(Int16.(got).-Int16.(expected[case]))));ok &=got==expected[case];append!(allgot,got)
        end
        allok &=ok;secs=total/(repeats*length(inputs));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"morphology_trial$(trial).bin");write(file,allgot)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"closing_cycle_four_separable_kernels_two_temporary_resets\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,string(ok),length(allgot),maxabs,repeats,gcsecs,gcpauses,file,length(allgot))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
