using CUDA
using Printf
function study_main(prepare,args=ARGS)
    groups=4;k=32;n=48;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat")
    ccall(:srand48,Cvoid,(Clong,),123);cdf=cumsum([.2+ccall(:drand48,Cdouble,()) for _ in 1:groups]);m=zeros(Int32,groups)
    for _ in 1:groups*8
        x=ccall(:drand48,Cdouble,())*last(cdf);g=1;while g<groups&&x>cdf[g];g+=1;end;m[g]+=1
    end
    m=max.(m,Int32(1));ccall(:srand48,Cvoid,(Clong,),123);a=Float16[];b=Float16[];oa=Int32[0];ob=Int32[0];oc=Int32[0];expected=Float32[];original_expected=Float32[]
    for g in 1:groups
        af=Float32[Float32(ccall(:drand48,Cdouble,())) for _ in 1:m[g]*k];bf=Float32[Float32(ccall(:drand48,Cdouble,())) for _ in 1:k*n];ag=Float16.(af);bg=Float16.(bf);append!(a,ag);append!(b,bg)
        for j in 0:n-1,i in 0:m[g]-1
            s=0f0;sf=0f0
            for q in 0:k-1
                s+=Float32(ag[i+q*m[g]+1])*Float32(bg[q+j*k+1]);sf+=af[i+q*m[g]+1]*bf[q+j*k+1]
            end
            push!(expected,Float32(Float16(s)));push!(original_expected,sf)
        end
        push!(oa,length(a));push!(ob,length(b));push!(oc,length(expected))
    end
    da,db,dc=CuArray(a),CuArray(b),CUDA.zeros(Float16,length(expected));launch! = prepare(da,db,dc,m,oa,ob,oc,k,n);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds=(time_ns()-t0)*1e-9/repeat;got=Float32.(Array(dc));err=maximum(abs.(got.-expected));origerr=maximum(abs.(got.-original_expected));ok=all(isfinite,got)&&err<=.02&&origerr<=1;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"groupgemm_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"original_fullfloat_reference_error\":%.12g,\"timing_region\":\"full_four_group_FP16_MoE_GEMM_FP32compute_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(got),err,origerr,repeat,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
