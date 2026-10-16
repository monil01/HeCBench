using CUDA
using Printf
const STUDY_W=256
const STUDY_H=256
const STUDY_POLE=Float32(sqrt(3f0)-2f0)
function initial_causal_ref(image, base::Int, data_length::Int, stride::Int)
    horizon = min(12, data_length)
    zn = STUDY_POLE
    sum = image[base]
    idx = base
    for _ in 0:(horizon - 1)
        sum += zn * image[idx]
        zn *= STUDY_POLE
        idx += stride
    end
    return sum
end

function convert_coefficients_ref!(image, base::Int, data_length::Int, stride::Int)
    lambda = (Float32(1.0) - STUDY_POLE) * (Float32(1.0) - Float32(1.0) / STUDY_POLE)
    idx = base
    previous = lambda * initial_causal_ref(image, base, data_length, stride)
    image[idx] = previous

    for _ in 2:data_length
        idx += stride
        previous = lambda * image[idx] + STUDY_POLE * previous
        image[idx] = previous
    end

    previous = (STUDY_POLE / (STUDY_POLE - Float32(1.0))) * image[idx]
    image[idx] = previous

    for _ in (data_length - 2):-1:0
        idx -= stride
        previous = STUDY_POLE * (previous - image[idx])
        image[idx] = previous
    end
    return image
end

function to_coef_2d_x_ref!(image, width::Int, height::Int)
    for y0 in 0:(height - 1)
        convert_coefficients_ref!(image, y0 * width + 1, width, 1)
    end
    return image
end

function to_coef_2d_y_ref!(image, width::Int, height::Int)
    for x0 in 0:(width - 1)
        convert_coefficients_ref!(image, x0 + 1, height, width)
    end
    return image
end

function study_main(prepare,args=ARGS)
    w=STUDY_W;h=STUDY_H;n=w*h;repeats=isempty(args) ? 20 : parse(Int,args[1]);repeats>0||error("positive repeats")
    input=Float32[Float32((i*37+17)%257-128)/128f0 for i in 0:n-1];expected=copy(input);to_coef_2d_x_ref!(expected,w,h);to_coef_2d_y_ref!(expected,w,h)
    image=CuArray(input);launch! = prepare(image,w,h);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        elapsed=0.0;gct=0.0;gcp=0
        for rep in 1:repeats
            copyto!(image,input);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();launch!();CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeats;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
        end
        got=Array(image);err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));ok=all(isfinite,got)&&ma<=1f-3;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"cbsfil_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"one_xy_recursive_prefilter_pair_reset_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,ma,mr,repeats,gct,gcp,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
