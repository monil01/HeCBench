using CUDA, Printf
function projectile_oracle(angle, velocity)
    n=length(angle);r=zeros(Float32,n);t=similar(r);h=similar(r)
    for i in 1:n
        s=sin(angle[i]*3.1415f0/180f0);c=cos(angle[i]*3.1415f0/180f0)
        t[i]=abs(2f0*velocity[i]*s)/9.81f0
        r[i]=abs(velocity[i]*t[i]*c)
        h[i]=(velocity[i]*velocity[i]*s*s)/2f0*9.81f0
    end
    vcat(r,t,h)
end
function projectile_study(launch!; args=ARGS)
    n=length(args)>0 ? parse(Int,args[1]) : 1048576
    repeats=length(args)>1 ? parse(Int,args[2]) : 10
    n>0 && n%256==0 && repeats>0 || error("fixture requires positive n divisible by 256")
    # Shared deterministic integer-valued Float32 input: original native operation.
    angle=Float32[10+(i%90) for i in 0:n-1]
    velocity=Float32[10+(i%400) for i in 0:n-1]
    expected=projectile_oracle(angle,velocity)
    a=CuArray(angle);v=CuArray(velocity)
    ranges=CUDA.zeros(Float32,n);times=similar(ranges);heights=similar(ranges)
    launch!(a,v,ranges,times,heights,Int32(n));CUDA.synchronize() # device JIT outside measurements
    allok=true
    for trial in 1:5
        CUDA.fill!(ranges,0f0);CUDA.fill!(times,0f0);CUDA.fill!(heights,0f0);CUDA.synchronize()
        t0=time_ns()
        for _ in 1:repeats
            launch!(a,v,ranges,times,heights,Int32(n))
        end
        CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeats
        actual=vcat(Array(ranges),Array(times),Array(heights))
        err=abs.(Float64.(actual).-Float64.(expected))
        maxabs=maximum(err);maxrel=maximum(err./max.(abs.(Float64.(expected)),Float64(eps(Float32))))
        ok=all(isfinite,actual) && maxabs<=1.0;allok &= ok
        output=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"projectile_$(trial).bin")
        open(output,"w") do io;write(io,actual);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(actual),maxabs,maxrel,repeats,output,length(actual))
    end
    println(allok ? "PASS" : "FAIL");return allok ? 0 : 1
end
