using CUDA
using Printf
const STUDY_RY = Int32(15)
const STUDY_YG = Int32(6)
const STUDY_GC_SEG = Int32(4)
const STUDY_CB = Int32(11)
const STUDY_BM = Int32(13)
const STUDY_MR = Int32(6)
const STUDY_MAXCOLS = Int32(55)

@inline function study_colorwheel_value(k::Int32, channel::Int32)
    if k < STUDY_RY
        i = k
        r = Int32(255)
        g = Int32(255) * i ÷ STUDY_RY
        b = Int32(0)
    elseif k < STUDY_RY + STUDY_YG
        i = k - STUDY_RY
        r = Int32(255) - Int32(255) * i ÷ STUDY_YG
        g = Int32(255)
        b = Int32(0)
    elseif k < STUDY_RY + STUDY_YG + STUDY_GC_SEG
        i = k - STUDY_RY - STUDY_YG
        r = Int32(0)
        g = Int32(255)
        b = Int32(255) * i ÷ STUDY_GC_SEG
    elseif k < STUDY_RY + STUDY_YG + STUDY_GC_SEG + STUDY_CB
        i = k - STUDY_RY - STUDY_YG - STUDY_GC_SEG
        r = Int32(0)
        g = Int32(255) - Int32(255) * i ÷ STUDY_CB
        b = Int32(255)
    elseif k < STUDY_RY + STUDY_YG + STUDY_GC_SEG + STUDY_CB + STUDY_BM
        i = k - STUDY_RY - STUDY_YG - STUDY_GC_SEG - STUDY_CB
        r = Int32(255) * i ÷ STUDY_BM
        g = Int32(0)
        b = Int32(255)
    else
        i = k - STUDY_RY - STUDY_YG - STUDY_GC_SEG - STUDY_CB - STUDY_BM
        r = Int32(255)
        g = Int32(0)
        b = Int32(255) - Int32(255) * i ÷ STUDY_MR
    end

    return channel == Int32(0) ? r : (channel == Int32(1) ? g : b)
end

@inline function study_compute_color_channel(fx::Float32, fy::Float32, out_channel::Int32)
    rad = sqrt(fx * fx + fy * fy)
    a = atan(-fy, -fx) / Float32(pi)
    fk = (a + 1.0f0) * 0.5f0 * Float32(STUDY_MAXCOLS - Int32(1))
    k0 = Int32(trunc(fk))
    k1 = k0 + Int32(1)
    if k1 == STUDY_MAXCOLS
        k1 = Int32(0)
    end
    f = fk - Float32(k0)

    cw_channel = Int32(2) - out_channel
    col0 = Float32(study_colorwheel_value(k0, cw_channel)) / 255.0f0
    col1 = Float32(study_colorwheel_value(k1, cw_channel)) / 255.0f0
    col = (1.0f0 - f) * col0 + f * col1
    if rad <= 1.0f0
        col = 1.0f0 - rad * (1.0f0 - col)
    else
        col *= 0.75f0
    end
    return UInt8(trunc(Int32, 255.0f0 * col))
end


function study_loop!(launch!,repeats)
    for _ in 1:repeats
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    size=512;half=size÷2;truerange=1f0;range=1.04f0*truerange;repeats=isempty(args) ? 100 : parse(Int,args[1]);repeats>0||error("positive repeats")
    expected=zeros(UInt8,3*size*size)
    for y in 0:size-1,x in 0:size-1
        if x!=half && y!=half
            fx=(Float32(x)/Float32(half)*range-range)/truerange;fy=(Float32(y)/Float32(half)*range-range)/truerange
            for ch in 0:2
                expected[(y*size+x)*3+ch+1]=study_compute_color_channel(fx,fy,Int32(ch))
            end
        end
    end
    pix=CUDA.zeros(UInt8,length(expected));launch! = prepare(pix,size,half,range,truerange);study_loop!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.fill!(pix,0x00);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_loop!(launch!,repeats);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num()
        got=Array(pix);err=abs.(Int.(got).-Int.(expected));ma=maximum(err);ok=ma<=1;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"colorwheel_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"colorwheel_kernel_only_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,repeats,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
