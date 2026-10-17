using CUDA
using Printf
function cpu_transform(x,dim,scale,T)
    gold=Float64.(x)
    for base in 0:dim:length(x)-1
        stride=1
        while stride<dim
            for block in 0:2stride:dim-1,j in 0:stride-1
                i=base+block+j+1;u=gold[i];v=gold[i+stride];gold[i]=u+v;gold[i+stride]=u-v
            end
            stride*=2
        end
    end
    Float32.(T.(gold.*Float64(scale))),gold.*Float64(scale)
end
function study_main(prepare,args=ARGS)
    batch=2;repeat=isempty(args) ? 30 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[];dense_max_error=0.0
    for dim in (8,64,512,4096,32768),T in (Float32,Float16,CUDA.CUDACore.BFloat16)
        x=T[Float32((i*17+row*3)%33-16)/32f0 for row in 0:batch-1 for i in 0:dim-1];scale=1f0/sqrt(Float32(dim));expected,gold=cpu_transform(x,dim,scale,T)
        if dim<=64
            for row in 0:batch-1,i in 0:dim-1
                direct=sum((isodd(count_ones(i&j)) ? -1.0 : 1.0)*Float64(x[row*dim+j+1]) for j in 0:dim-1)*Float64(scale)
                dense_max_error=max(dense_max_error,abs(direct-gold[row*dim+i+1]))
            end
        end
        dx=CuArray(x);out=CUDA.zeros(T,length(x));a=CUDA.zeros(Float32,length(x));bb=similar(a);launch! = prepare(dx,out,a,bb,dim,scale);launch!();push!(fixtures,(out,launch!,expected,gold,x,scale,dim,string(T)))
    end
    dense_max_error==0||error("dense Walsh sign-matrix witness");CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];seconds=0.0;ma=0.0;energy_error=0.0;ok=true;casetimes=Float64[]
        for (out,launch!,expected,gold,x,scale,dim,kind) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();secondsone=(time_ns()-t0)*1e-9/repeat;seconds+=secondsone;push!(casetimes,secondsone);got=Float32.(Array(out));err=maximum(abs.(got.-expected));ma=max(ma,err)
            # Energy compares rounded output to exact normalization including native Float32 scale.
            ideal=sum(abs2,Float64.(x))*dim*Float64(scale)^2;ee=abs(sum(abs2,Float64.(got))-ideal)/max(ideal,eps());energy_error=max(energy_error,ee)
            ok &=all(isfinite,got)&&err<=1e-3&&ee<0.02;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"hadamard_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"energy_relative_error\":%.12g,\"dense_sign_oracle_max_error\":%.12g,\"timing_region\":\"sum_15_complete_normalized_Walsh_Hadamard_transforms\",\"jit_excluded\":true,\"internal_repeats\":%d,\"case_times_seconds\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),ma,energy_error,dense_max_error,repeat,string(casetimes),file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
