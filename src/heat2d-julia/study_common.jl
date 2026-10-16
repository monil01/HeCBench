# Shared study fixture/oracle/measurement adapter; original CLI remains preserved.
using CUDA
using Printf
function cpu_step!(out, inp, delta, norm, Lx, Ly)
    @inbounds for y in 0:Ly-1, x in 0:Lx-1
        i=y*Lx+x+1
        out[i]=norm*inp[i]+delta*(inp[y*Lx+mod(x+1,Lx)+1]+inp[y*Lx+mod(x-1,Lx)+1]+inp[mod(y+1,Ly)*Lx+x+1]+inp[mod(y-1,Ly)*Lx+x+1])
    end
end
function run_steps!(launch!, a, b, niter)
    for _ in 1:niter
        launch!(b,a)
        a,b=b,a
    end
    return a,b
end
function study_main(prepare, args=ARGS)
    Lx=length(args)>=1 ? parse(Int,args[1]) : 512
    Ly=length(args)>=2 ? parse(Int,args[2]) : 512
    niter=length(args)>=3 ? parse(Int,args[3]) : 100
    Lx>0 && Ly>0 && Lx%16==0 && Ly%16==0 && niter>0 || error("positive dimensions multiple16 and positive niter required")
    n=Lx*Ly
    input=Float32[Float32(mod((i-1)*13+7,101))/100f0 for i in 1:n]
    sigma=0.01f0;delta=sigma/(1f0+4f0*sigma);norm=1f0/(1f0+4f0*sigma)
    expected=copy(input);tmp=similar(expected)
    for _ in 1:niter
        cpu_step!(tmp,expected,delta,norm,Lx,Ly);expected,tmp=tmp,expected
    end
    a=CuArray(input);b=CuArray(input)
    launch! = prepare(a,b,Lx,Ly,delta,norm)
    run_steps!(launch!,a,b,1);CUDA.synchronize()
    allok=true
    for trial in 1:5
        copyto!(a,input);copyto!(b,input);CUDA.synchronize()
        t0=time_ns()
        a,b=run_steps!(launch!,a,b,niter)
        CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/niter
        got=Array(a);err=abs.(got.-expected);maxabs=maximum(err)
        maxrel=maximum(err./max.(abs.(expected),eps(Float32)))
        ok=all(isfinite,got) && all(isapprox.(got,expected;atol=1f-5,rtol=1f-5));allok &= ok
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"heat2d_trial$(trial).bin");write(artifact,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,maxabs,maxrel,niter,artifact,n)
    end
    println(allok ? "PASS" : "FAIL");return allok ? 0 : 1
end
