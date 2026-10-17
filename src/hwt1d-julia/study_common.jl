using CUDA
using Printf
function haar_reference(input::Vector{Float32})
    n = length(input)
    temp = input ./ sqrt(Float32(n))
    out = zeros(Float32, n)
    len = n
    while len > 1
        @inbounds for i0 in 0:(len ÷ 2 - 1)
            data0 = temp[2 * i0 + 1]
            data1 = temp[2 * i0 + 2]
            out[i0 + 1] = (data0 + data1) / sqrt(2.0f0)
            out[len ÷ 2 + i0 + 1] = (data0 - data1) / sqrt(2.0f0)
        end
        copyto!(temp, out)
        len >>= 1
    end
    return out
end


function study_transform!(scale!,level!,data,scratch,n)
    scale!();len=n
    while len>1
        level!(Int32(len));copyto!(view(data,1:len),view(scratch,1:len));len>>=1
    end
end
function study_main(prepare,args=ARGS)
    n=16384;repeats=isempty(args) ? 20 : parse(Int,args[1]);repeats>0||error("positive repeats");ccall(:srand,Cvoid,(Cuint,),Cuint(2));input=Float32[Float32(ccall(:rand,Cint,())%10) for _ in 1:n];expected=haar_reference(input)
    data=CuArray(input);scratch=CUDA.zeros(Float32,n);scale!,level! = prepare(data,scratch,n);study_transform!(scale!,level!,data,scratch,n);CUDA.synchronize();allok=true
    for trial in 1:5
        elapsed=0.0;gct=0.0;gcp=0
        for rep in 1:repeats
            copyto!(data,input);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_transform!(scale!,level!,data,scratch,n);CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
        end
        got=Array(data);err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));ok=all(isfinite,got)&&ma<=.1f0;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"hwt1d_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"full_device_transform_including_normalization_and_required_d2d_copies\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,mr,repeats,gct,gcp,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
