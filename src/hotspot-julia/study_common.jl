using CUDA
using Printf
const STUDY_L=512
function study_constants()
    grid=.016f0/Float32(STUDY_L)
    cap=(((.5f0*1.75f6)*.0005f0)*grid)*grid
    rx=grid/(((2f0*100f0)*.0005f0)*grid)
    ry=rx;rz=.0005f0/((100f0*grid)*grid)
    step=.001f0/(3f6/((.5f0*.0005f0)*1.75f6))
    cap,rx,ry,rz,step
end
function cpu_step!(out,cur,power,dc,rx,ry,rz)
    L=STUDY_L
    @inbounds for y in 1:L,x in 1:L
        n=max(y-1,1);s=min(y+1,L);w=max(x-1,1);e=min(x+1,L);i=(y-1)*L+x;t=cur[i]
        out[i]=t+dc*(power[i]+(cur[(s-1)*L+x]+cur[(n-1)*L+x]-2f0*t)*ry+(cur[(y-1)*L+e]+cur[(y-1)*L+w]-2f0*t)*rx+(80f0-t)*rz)
    end
end
function run_steps!(launch!,a,b,steps)
    for _ in 1:steps
        launch!(b,a);a,b=b,a
    end
    a,b
end
function study_main(prepare,args=ARGS)
    steps=length(args)>=1 ? parse(Int,args[1]) : 200
    steps>0||error("positive stepsrequired");n=STUDY_L*STUDY_L
    input=Float32[300f0+Float32((i-1)%101)*.5f0 for i in 1:n]
    power=Float32[Float32((i-1)%17)*.03125f0 for i in 1:n]
    cap,rx,ry,rz,step=study_constants();dc=step/cap;rx1=1f0/rx;ry1=1f0/ry;rz1=1f0/rz
    expected=copy(input);tmp=similar(expected)
    for _ in 1:steps
        cpu_step!(tmp,expected,power,dc,rx1,ry1,rz1);expected,tmp=tmp,expected
    end
    a=CuArray(input);b=CuArray(input);p=CuArray(power)
    launch! = prepare(a,b,p,dc,rx1,ry1,rz1)
    run_steps!(launch!,a,b,1);CUDA.synchronize();allok=true
    for trial in 1:5
        copyto!(a,input);copyto!(b,input);CUDA.synchronize()
        t0=time_ns();a,b=run_steps!(launch!,a,b,steps);CUDA.synchronize();secs=(time_ns()-t0)*1e-9/steps
        got=Array(a);err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)))
        ok=all(isfinite,got)&&ma<=1f-3;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"hotspot_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,string(ok),n,ma,mr,steps,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
