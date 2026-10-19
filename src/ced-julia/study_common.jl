# Shared host harness: each framework defines allocate_model and launch_stage!.
function make_work(s)
    initial=read(joinpath(@__DIR__,s.input));length(initial)==s.n||error("input extent")
    buffers=[vcat(fill(UInt8(211),16),x,fill(UInt8(211),16)) for x in (initial,zeros(UInt8,s.n),zeros(UInt8,s.n))]
    (;s,buffers,device=map(allocate_model,buffers))
end
function reset_work!(w)
    for k=1:3
        copyto!(w.device[k],w.buffers[k])
    end
end
function capture_work!(a,w)
    for d in w.device
        append!(a,Int32.(Array(d)))
    end
end
function coefficients!(a,coeff)
    append!(a,reinterpret(Int32,Array(coeff[1])));append!(a,Array(coeff[2]));append!(a,Array(coeff[3]))
end
function check_output(a,expected;trace=false)
    length(a)==length(expected)||return (false,typemax(Int))
    good=true;maximum=0;off=0
    if !trace
        for i=1:PREFIX
            d=abs(Int(a[i])-Int(expected[i]));maximum=max(maximum,d);good&=d==0
        end
        off=PREFIX
    end
    for s in CASES,k=1:(trace ? 5 : 3)
        for j=1:s.n+32
            d=abs(Int(a[off+j])-Int(expected[off+j]));maximum=max(maximum,d)
            budget=(trace ? (k==2||k==4) : k==2)&&17<=j<=s.n+16 ? 1 : 0
            good&=d<=budget
        end
        off+=s.n+32
    end
    (good,maximum)
end
function study_main(model)
    expected=reinterpret(Int32,read(ARGS[1]));trace_expected=reinterpret(Int32,read(ARGS[2]));tracedir=ARGS[3];mkpath(tracedir)
    coeff=(allocate_model(GAUSS),allocate_model(SOBX),allocate_model(SOBY));works=map(make_work,CASES)
    pipeline!()=begin
        for w in works,k=1:4
            launch_stage!(k,w,coeff)
        end
        CUDA.device_synchronize()
    end
    pipeline!() # exact all-kernel GPU/host specialization warmup outside five trials
    tracepath=joinpath(tracedir,"untimed_trace_$(getpid()).bin")
    for trial=1:5
        foreach(reset_work!,works)
        actual=Int32[];coefficients!(actual,coeff)
        for w in works
            s=w.s;append!(actual,Int32[s.caseid,s.frame,s.rows,s.cols,TILE]);capture_work!(actual,w)
        end
        CUDA.device_synchronize();start=time_ns();pipeline!();CUDA.device_synchronize();seconds=(time_ns()-start)/1e9
        coefficients!(actual,coeff);length(actual)==PREFIX||error("prefix extent")
        foreach(w->capture_work!(actual,w),works)
        good,err=check_output(actual,expected);out=joinpath(ENV["STUDY_OUTPUT_DIR"],"ced_trial$(trial).bin");write(out,actual)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$good,\"checked_values\":$(length(actual)),\"max_abs_error\":$err,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"resident12frames_48source_kernel_launches_whole_device_completion\",\"internal_repeats\":1,\"iterations\":1,\"frames\":12,\"kernel_launches\":48,\"supplemental_trace_path\":\"$tracepath\",\"output_artifact\":{\"path\":\"$out\",\"dtype\":\"int32\",\"count\":$(length(actual)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("independent CPU full-output contract failed")
    end
    trace=Int32[]
    for w in works
        reset_work!(w)
        launch_stage!(1,w,coeff);append!(trace,Int32.(Array(w.device[2])))
        launch_stage!(2,w,coeff);append!(trace,Int32.(Array(w.device[1])));append!(trace,Int32.(Array(w.device[3])))
        launch_stage!(3,w,coeff);append!(trace,Int32.(Array(w.device[2])))
        launch_stage!(4,w,coeff);append!(trace,Int32.(Array(w.device[1])))
    end
    CUDA.device_synchronize();good,err=check_output(trace,trace_expected;trace=true);write(tracepath,trace)
    println("STUDY_AUXILIARY untimed_complete_stage_trace correct=$good checked_values=$(length(trace)) max_abs_error=$err path=$tracepath")
    good||error("untimed full-stage oracle failed")
end
