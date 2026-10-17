using CUDA
using Printf
function study_main(prepare,args=ARGS)
    m=64;k=32;n=48;repeat=isempty(args) ? 10 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    for T in (Float16,Float32,Float64)
        ccall(:srand,Cvoid,(Cuint,),Cuint(2));a=T[T(ccall(:rand,Cint,())%2) for _ in 1:m*k];b=T[T(ccall(:rand,Cint,())%2) for _ in 1:k*n];c0=T[T(ccall(:rand,Cint,())%2) for _ in 1:m*n];expected=zeros(Float64,m*n)
        for row in 0:m-1,col in 0:n-1
            s=0.0
            for q in 0:k-1
                s+=Float64(a[row*k+q+1])*Float64(b[q*n+col+1])
            end
            expected[row*n+col+1]=2.0*s+.5*Float64(c0[row*n+col+1])
        end
        da,db,dc=CuArray(a),CuArray(b),CuArray(c0);launch! = prepare(da,db,dc,m,k,n);launch!();push!(fixtures,(dc,c0,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float64[];times=Float64[];ma=0.0;ok=true;gct=0.0;gcp=0
        for (dc,c0,launch!,expected) in fixtures
            elapsed=0.0
            for rep in 1:repeat
                copyto!(dc,c0);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();launch!();CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
            end
            push!(times,elapsed);got=Float64.(Array(dc));err=maximum(abs.(got.-expected));ma=max(ma,err);ok &=all(isfinite,got)&&err==0;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"blas_gemm_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_FP16_FP32_FP64_GEMM_alpha2_betaHalf_one_operation_reset_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"dtype_times_seconds\":[%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeat,times[1],times[2],times[3],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
