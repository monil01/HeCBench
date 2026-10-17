using CUDA
using Printf
function study_main(prepare,args=ARGS)
    rows=32;terms=64;cols=48;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");ccall(:srand,Cvoid,(Cuint,),123);b=Float32[Float32(ccall(:rand,Cint,())%256) for _ in 1:terms*cols]
    for col in 0:cols-1
        s=0f0;for i in 0:terms-1;s+=b[i*cols+col+1];end;for i in 0:terms-1;b[i*cols+col+1]/=s;end
    end
    db=CuArray(b);fixtures=[]
    for kind in 0:1
        a=kind==0 ? fill(1f0/Float32(terms),rows*terms) : Float32[Float32((row*13+i*7)%17+1) for row in 0:rows-1 for i in 0:terms-1]
        if kind==1
            for row in 0:rows-1
                s=0f0;for i in 0:terms-1;s+=a[row*terms+i+1];end;for i in 0:terms-1;a[row*terms+i+1]/=s;end
            end
        end
        expected=zeros(Float32,rows*cols);gold=zeros(Float64,rows*cols)
        for row in 0:rows-1,col in 0:cols-1
            s=0f0;ds=0.0
            for i in 0:terms-1
                av=a[row*terms+i+1];bv=b[i*cols+col+1];s+=sqrt(av*bv);ds+=sqrt(Float64(av)*Float64(bv))
            end
            value=1f0-s;expected[row*cols+col+1]=sqrt((signbit(value) ? 0f0 : 1f0)*value);gold[row*cols+col+1]=sqrt(max(0.,1-ds))
        end
        da,dc=CuArray(a),CUDA.zeros(Float32,rows*cols);launch! = prepare(da,db,dc,rows,terms,cols);launch!();push!(fixtures,(dc,launch!,expected,gold))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ok=true;ma=0.0;oracleerror=0.0
        for (dc,launch!,expected,gold) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);got=Array(dc);err=maximum(abs.(got.-expected));de=maximum(abs.(Float64.(got).-gold));ma=max(ma,err);oracleerror=max(oracleerror,de);ok &=all(isfinite,got)&&all(0 .<=got.<=1)&&err<1e-5&&de<1e-5;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"hellinger_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"float64_oracle_max_error\":%.12g,\"timing_region\":\"sum_uniform_and_nonuniform_normalized_probability_hellinger_matrices\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,oracleerror,repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
