using CUDA
using Printf
function study_input(n)
    data=Vector{Float32}(undef,n);s=UInt32(1)
    for i in 1:n
        s=s*UInt32(1103515245)+UInt32(12345)
        data[i]=Float32((s>>8)&UInt32(0xffffff))/16777216f0
    end
    data
end
function cpu_step!(out,inp,coeff,nx,ny,nz,r)
    ox=nx+2r;oy=ny+2r;oz=ox*oy
    @inbounds for z in 1:nz,y in 1:ny,x in 1:nx
        i=x+r+(y+r-1)*ox+(z+r-1)*oz
        value=coeff[1]*inp[i]
        for k in 1:r
            c=coeff[k+1]
            value+=c*(inp[i+k]+inp[i-k])
            value+=c*(inp[i+k*ox]+inp[i-k*ox])
            value+=c*(inp[i+k*oz]+inp[i-k*oz])
        end
        out[i]=value
    end
end
function run_steps!(launch!,a,b,steps)
    for _ in 1:steps
        launch!(b,a);a,b=b,a
    end
    a,b
end
function study_main(prepare,args=ARGS)
    nx=length(args)>=1 ? parse(Int,args[1]) : 96
    ny=length(args)>=2 ? parse(Int,args[2]) : 96
    nz=length(args)>=3 ? parse(Int,args[3]) : 96
    steps=length(args)>=4 ? parse(Int,args[4]) : 5
    nx>=96 && ny>=96 && nz>=96 && nx%32==0 && ny%8==0 && steps>0 || error("dimensions>=96, xmultiple32 ymultiple8, positive steps")
    r=4;n=(nx+2r)*(ny+2r)*(nz+2r)
    input=study_input(n);coeff=fill(.1f0,r+1)
    expected=copy(input);temp=copy(input)
    for _ in 1:steps
        cpu_step!(temp,expected,coeff,nx,ny,nz,r);expected,temp=temp,expected
    end
    a=CuArray(input);b=CuArray(input);c=CuArray(coeff)
    launch! = prepare(a,b,c,nx,ny,nz,r)
    run_steps!(launch!,a,b,1);CUDA.synchronize()
    allok=true
    for trial in 1:5
        copyto!(a,input);copyto!(b,input);CUDA.synchronize()
        t0=time_ns();a,b=run_steps!(launch!,a,b,steps);CUDA.synchronize()
        elapsed=(time_ns()-t0)*1e-9/steps
        got=Array(a);err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)))
        ok=all(isfinite,got)&&all(isapprox.(got,expected;atol=1f-5,rtol=1f-4));allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"fdtd3d_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,ma,mr,steps,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
