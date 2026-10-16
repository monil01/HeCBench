using CUDA
using Printf
const STUDY_N=2048
function study_input()
    n=STUDY_N;v=zeros(Float32,n*n)
    for j in 0:n-1,i in 0:n-1
        if i==0||i==n-1||j==0||j==n-1
            v[i+j*n+1]=Float32((i+3j)%17)/16f0
        end
    end
    v
end
function cpu_oracle(input,steps)
    n=STUDY_N;a=copy(input);b=copy(input);norms=zeros(Float32,steps)
    for k in 1:steps
        total=0.0
        @inbounds for j in 2:n-1,i in 2:n-1
            idx=i+(j-1)*n
            value=.25f0*(a[idx+1]+a[idx-1]+a[idx+n]+a[idx-n]);b[idx]=value
            df=value-a[idx];total+=Float64(df*df)
        end
        norms[k]=Float32(sqrt(total/(n*n)));a,b=b,a
    end
    a,norms
end
function run_steps!(launch!,a,b,err,steps)
    norms=zeros(Float32,steps)
    for k in 1:steps
        CUDA.fill!(err,0f0)
        launch!(b,a,err)
        a,b=b,a
        e=Array(err)[1] # Original convergence loop's per-step synchronization/copy retained.
        norms[k]=sqrt(e/Float32(STUDY_N*STUDY_N))
    end
    a,b,norms
end
function study_main(prepare,args=ARGS)
    steps=length(args)>=1 ? parse(Int,args[1]) : 20
    steps>0||error("positive stepsrequired");n=STUDY_N*STUDY_N;input=study_input()
    expected,expectednorms=cpu_oracle(input,steps)
    a=CuArray(input);b=CuArray(input);err=CUDA.zeros(Float32,1);launch! = prepare(a,b,err)
    run_steps!(launch!,a,b,err,1);CUDA.synchronize();allok=true
    for trial in 1:5
        copyto!(a,input);copyto!(b,input);CUDA.fill!(err,0f0);CUDA.synchronize()
        gc_before=Base.gc_num()
        t0=time_ns();a,b,norms=run_steps!(launch!,a,b,err,steps);CUDA.synchronize();secs=(time_ns()-t0)*1e-9/steps
        gc_after=Base.gc_num();gc_secs=(gc_after.total_time-gc_before.total_time)*1e-9;gc_pauses=gc_after.pause-gc_before.pause
        got=Array(a);allgot=vcat(got,norms);allref=vcat(expected,expectednorms)
        diff=abs.(allgot.-allref);ma=maximum(diff);mr=maximum(diff./max.(abs.(allref),eps(Float32)))
        ok=all(isfinite,allgot)&&all(isapprox.(allgot,allref;atol=1f-6,rtol=1f-4));allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"jacobi_trial$(trial).bin");write(file,allgot)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"fixed_jacobi_step_loop_with_error_reset_reduction_copy_per_step\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,string(ok),length(allgot),ma,mr,steps,gc_secs,gc_pauses,file,length(allgot))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
