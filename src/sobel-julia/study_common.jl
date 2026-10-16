using CUDA
using Printf
@inline function study_clamp_u8(v::Float32)
    if v > 255.0f0
        return UInt8(255)
    elseif v < 0.0f0
        return UInt8(0)
    else
        return UInt8(unsafe_trunc(Int32, v))
    end
end

function study_cpu!(inp::Vector{UInt8}, out::Vector{UInt8}, width, height)
    for y in 1:(height-2)
        for x in 1:(width-2)  # 0-based: x=1..width-2, y=1..height-2
            c = x + y * width  # 0-based pixel idx
            for ch in 0:3
                function px(off, ch)
                    idx = 4 * (c + off) + ch  # 0-based
                    return Float32(inp[idx + 1])
                end
                i00 = px(-1 - width, ch); i01 = px(-width, ch); i02 = px(1 - width, ch)
                i10 = px(-1, ch);                             i12 = px(1, ch)
                i20 = px(-1 + width, ch); i21 = px(width, ch); i22 = px(1 + width, ch)
                Gx = i00 + 2f0 * i10 + i20 - i02 - 2f0 * i12 - i22
                Gy = i00 - i20 + 2f0 * i01 - 2f0 * i21 + i02 - i22
                mag = sqrt(Gx*Gx + Gy*Gy) / 2f0
                out_idx = 4 * c + ch + 1
                out[out_idx] = study_clamp_u8(mag)
            end
        end
    end
end

function study_loop!(launch!, repeats)
    for _ in 1:repeats
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    w=512;h=512;repeats=isempty(args) ? 100 : parse(Int,args[1]);repeats>0||error("positive repeats")
    input=Vector{UInt8}(undef,4*w*h);state=UInt64(0x1234567)
    for i in eachindex(input)
        state=state*UInt64(6364136223846793005)+UInt64(1442695040888963407);input[i]=UInt8((state>>32)&UInt64(255))
    end
    expected=zeros(UInt8,length(input));study_cpu!(input,expected,w,h)
    din=CuArray(input);dout=CUDA.zeros(UInt8,length(input));launch! = prepare(din,dout,w,h)
    study_loop!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.fill!(dout,0x00);CUDA.synchronize();gb=Base.gc_num();t0=time_ns()
        study_loop!(launch!,repeats);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num()
        got=Array(dout);errs=abs.(Int.(got).-Int.(expected));ma=maximum(errs);ok=ma==0;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"sobel_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sobel_kernel_only_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,repeats,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
