using CUDA
using Printf
const StudyBFloat16=CUDA.CUDACore.BFloat16
function study_reference(prefix_output, suffix_output, prefix_lse, suffix_lse,
                   num_tokens::Int, num_heads::Int, head_size::Int, ::Type{T}) where {T}
    total = num_tokens * num_heads * head_size
    output = Vector{T}(undef, total)
    lse = Vector{Float32}(undef, num_tokens * num_heads)
    for t in 0:(num_tokens - 1), h in 0:(num_heads - 1)
        lse_idx = t * num_heads + h + 1
        p_lse = prefix_lse[lse_idx]
        s_lse = suffix_lse[lse_idx]
        p_lse = isinf(p_lse) ? -Inf32 : p_lse
        s_lse = isinf(s_lse) ? -Inf32 : s_lse
        max_lse = max(p_lse, s_lse)
        p_exp = exp(p_lse - max_lse)
        s_exp = exp(s_lse - max_lse)
        out_se = p_exp + s_exp
        lse[lse_idx] = log(out_se) + max_lse
        p_scale = p_exp / out_se
        s_scale = s_exp / out_se
        base = (t * num_heads + h) * head_size
        for d in 1:head_size
            output[base + d] = T(Float32(prefix_output[base + d]) * p_scale +
                                 Float32(suffix_output[base + d]) * s_scale)
        end
    end
    return output, lse
end


function study_loop!(launch!,repeat)
    for _ in 1:repeat
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    tokens=128;heads=8;size=64;n=tokens*heads*size;ln=tokens*heads;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    pl=Float32[Float32((13i+7)%129-64)/16f0 for i in 0:ln-1];sl=Float32[Float32((37i+11)%129-64)/16f0 for i in 0:ln-1]
    for i in 1:32
        pl[i]=isodd(i) ? Inf32 : -Inf32;sl[32+i]=isodd(i) ? Inf32 : -Inf32
    end
    for T in (Float32,Float16,StudyBFloat16)
        prefix=T[T(Float32((13i+7)%129-64)/128f0) for i in 0:n-1];suffix=T[T(Float32((37i+11)%129-64)/128f0) for i in 0:n-1];ro,rl=study_reference(prefix,suffix,pl,sl,tokens,heads,size,T);expected=vcat(Float32.(ro),rl)
        dp,ds,dpl,dsl=CuArray(prefix),CuArray(suffix),CuArray(pl),CuArray(sl);out=CUDA.zeros(T,n);lse=CUDA.zeros(Float32,ln);launch! = prepare(out,dp,ds,lse,dpl,dsl,tokens,heads,size);study_loop!(launch!,1);push!(fixtures,(out,lse,launch!,expected))
        @printf("STUDY_DTYPE {\"name\":\"%s\",\"sizeof\":%d,\"compute\":\"float32\"}\n",string(T),sizeof(T))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];times=Float64[];ma=0f0;mr=0f0;ok=true;gct=0.0;gcp=0
        for (out,lse,launch!,expected) in fixtures
            CUDA.fill!(out,zero(eltype(out)));CUDA.fill!(lse,NaN32);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeat);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();push!(times,elapsed);gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;got=vcat(Float32.(Array(out)),Array(lse));err=abs.(got.-expected);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(expected),eps(Float32))));ok &=all(isfinite,got)&&maximum(err)<=1f-3;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"attention_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_real_FP32_FP16_BF16_attention_merge_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"dtype_times_seconds\":[%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,mr,repeat,times[1],times[2],times[3],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
