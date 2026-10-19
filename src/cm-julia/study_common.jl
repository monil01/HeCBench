readtyped(type,path)=copy(reinterpret(type,read(joinpath(@__DIR__,path))))
guard(vals,v)=vcat(fill(eltype(vals)(v),16),vals,fill(eltype(vals)(v),16))
function make_work(group)
    set=group==0 ? 2 : 1
    initial=(ref=guard(readtyped(Int32,"ref$(group)_i32.bin"),1234567),q=guard(readtyped(Int32,"query_i32.bin"),1234567),random=guard(readtyped(Float32,"random$(group)_f32.bin"),-999),dot=guard(zeros(Int32,DB),-12345),scores=guard(zeros(Float32,R),-999),count=guard(zeros(Int32,SB),-12345))
    device=map(allocate_model,initial)
    (;group,set,initial,device,hd=zeros(Int32,DB),hc=zeros(Int32,SB),stats=zeros(Float64,7))
end
function reset_work!(w)
    for key in keys(w.initial)
        copyto!(getproperty(w.device,key),getproperty(w.initial,key))
    end
end
function prefix!(out,w)
    append!(out,Float64[N,R,SIG,w.set,UC,DB,SB])
    for key in (:ref,:q,:random,:dot,:scores,:count)
        append!(out,Float64.(Array(getproperty(w.device,key))))
    end
end
function source_pvalue(above,n)
    left=0.0;right=0.0;p=0.0
    if above<1
        left=.5/n;right=(n-.5)/n
    elseif above>n-1
        right=.5/n;left=(n-.5)/n
    else
        left=above/n;right=(n-above)/n
    end
    if right<left;p=2right;end
    if right>left;p=2left;end
    p # Source exacthalf bug deliberately preserved; primary oddR excludes tie.
end
function work_pipeline!(w)
    launch_dot!(w)
    copyto!(w.hd,view(w.device.dot,17:16+DB))
    dot=0.0
    for x in w.hd;dot+=x;end
    avg=(dot/UC)/w.set;threshold=Float32(avg)
    launch_scores!(w);launch_count!(w,threshold)
    copyto!(w.hc,view(w.device.count,17:16+SB))
    above=0.0
    for x in w.hc;above+=x;end
    pv=source_pvalue(above,R);enfp=pv*1309
    w.stats .= (dot,avg,Float64(threshold),above,pv,enfp,Float64(enfp<1))
end
function tail!(out,w)
    for key in (:dot,:scores,:count)
        append!(out,Float64.(Array(getproperty(w.device,key))))
    end
    append!(out,w.stats)
end
function study_main(model)
    expected=readtyped(Float64,"oracle_f64.bin");works=map(make_work,0:1)
    pipeline!()=begin
        foreach(work_pipeline!,works);CUDA.device_synchronize()
    end
    pipeline!() # Actual exactsixGPUkernels plushostfeedback warmup outside five trials.
    println("STUDY_AUXILIARY source_pvalue_exacthalf=$(source_pvalue(50000.0,100000)) canonical=1; primaryoddR=$R")
    for trial=1:5
        foreach(reset_work!,works);out=Float64[];foreach(w->prefix!(out,w),works);length(out)==PREFIX||error("prefix extent")
        CUDA.device_synchronize();start=time_ns();pipeline!();CUDA.device_synchronize();seconds=(time_ns()-start)/1e9
        foreach(w->tail!(out,w),works);length(out)==COUNT||error("output extent")
        good=all(isfinite,out)&&out==expected;err=maximum(abs.(out.-expected));path=joinpath(ENV["STUDY_OUTPUT_DIR"],"cm_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$good,\"checked_values\":$(length(out)),\"max_abs_error\":$err,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"resident_two_drugsets_3GPUstages_dot_hostfeedback_count_hostpvalue\",\"internal_repeats\":1,\"iterations\":1,\"kernel_launches\":6,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float64\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("IndependentCPU fullscore/partials/statistics contract failed")
    end
end
