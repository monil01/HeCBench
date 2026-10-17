using CUDA
using Printf
function cpu_step(x,nx,ny,nz,T)
    out=similar(x);sixth=T(1)/T(6)
    for z in 0:nz-1,y in 0:ny-1,xx in 0:nx-1
        i=xx+y*nx+z*nx*ny+1
        out[i]=(xx==0||xx==nx-1||y==0||y==ny-1||z==0||z==nz-1) ? x[i] : (x[i-1]+x[i+1]+x[i-nx]+x[i+nx]+x[i-nx*ny]+x[i+nx*ny])*sixth
    end
    out
end
function study_main(prepare,args=ARGS)
    nx=64;ny=19;nz=17;steps=isempty(args) ? 20 : parse(Int,args[1]);steps==20||error("named20steps");fixtures=[]
    for kind in 0:1
        x=Float32[(kind==0 ? ((xx==0||xx==nx-1||y==0||y==ny-1||z==0||z==nz-1) ? 1f0 : 0f0) : Float32(xx+2y+3z)/256f0) for z in 0:nz-1 for y in 0:ny-1 for xx in 0:nx-1]
        expected=copy(x);gold=Float64.(x)
        for step in 1:steps;expected=cpu_step(expected,nx,ny,nz,Float32);gold=cpu_step(gold,nx,ny,nz,Float64);end
        a=CuArray(x);bb=similar(a);launch! = prepare(nx,ny,nz);launch!(a,bb);push!(fixtures,(x,a,bb,launch!,expected,gold,kind))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];seconds=0.0;ma=0.0;de=0.0;boundary_error=0.0;harmonic_error=0.0;ok=true
        for (x,a,bb,launch!,expected,gold,kind) in fixtures
            copyto!(a,x);CUDA.synchronize();src=a;dst=bb;t0=time_ns()
            for step in 1:steps;launch!(src,dst);src,dst=dst,src;end
            CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/steps;got=Array(src);ma=max(ma,maximum(abs.(got.-expected)));de=max(de,maximum(abs.(Float64.(got).-gold)));ok &=all(isfinite,got)&&all(abs.(got.-expected).<=1e-5)&&all(abs.(Float64.(got).-gold).<=1e-5)&&all(0f0.<=got.<=1f0)
            for z in 0:nz-1,y in 0:ny-1,xx in 0:nx-1
                i=xx+y*nx+z*nx*ny+1
                if xx==0||xx==nx-1||y==0||y==ny-1||z==0||z==nz-1;boundary_error=max(boundary_error,abs(got[i]-x[i]));end
            end
            kind==1&&(harmonic_error=max(harmonic_error,maximum(abs.(got.-x))));append!(output,got)
        end
        ok &=boundary_error==0&&harmonic_error<=1e-5;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"laplace3d_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"float64_oracle_max_error\":%.12g,\"boundary_max_error\":%.12g,\"harmonic_fixedpoint_error\":%.12g,\"timing_region\":\"sum_two_20step_relaxations_normalized_per_step\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),ma,de,boundary_error,harmonic_error,steps,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
