using CUDA
using Printf
const ST_KSIZE_H=Int32(11)
const ST_KSIZE_W=Int32(11)
const ST_STRIDE_H=Int32(4)
const ST_STRIDE_W=Int32(4)
const ST_PAD_H=Int32(1)
const ST_PAD_W=Int32(1)
function study_reference!(nthreads::Int, output_grad, channels::Int, input_h::Int, input_w::Int,
                    output_h::Int, output_w::Int, input_grad)
    for index0 in 0:nthreads-1
        w_offset = index0 % input_w + Int(ST_PAD_W)
        tmp = index0 ÷ input_w
        h_offset = tmp % input_h + Int(ST_PAD_H)
        tmp ÷= input_h
        offset_c = tmp % channels
        batch_idx = tmp ÷ channels

        phstart = h_offset < Int(ST_KSIZE_H) ? 0 : (h_offset - Int(ST_KSIZE_H)) ÷ Int(ST_STRIDE_H) + 1
        pwstart = w_offset < Int(ST_KSIZE_W) ? 0 : (w_offset - Int(ST_KSIZE_W)) ÷ Int(ST_STRIDE_W) + 1
        phend = min(h_offset ÷ Int(ST_STRIDE_H) + 1, output_h)
        pwend = min(w_offset ÷ Int(ST_STRIDE_W) + 1, output_w)

        gradient = 0.0f0
        output_stride = batch_idx * output_h * output_w * channels +
                        offset_c * output_h * output_w
        for ph in phstart:phend-1, pw in pwstart:pwend-1
            hstart = ph * Int(ST_STRIDE_H) - Int(ST_PAD_H)
            wstart = pw * Int(ST_STRIDE_W) - Int(ST_PAD_W)
            hend = min(hstart + Int(ST_KSIZE_H), input_h)
            wend = min(wstart + Int(ST_KSIZE_W), input_w)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            pool_size = (hend - hstart) * (wend - wstart)
            output_sub_idx = ph * output_w + pw
            gradient += (1.0f0 / Float32(pool_size)) * output_grad[output_stride + output_sub_idx + 1]
        end
        input_grad[index0 + 1] = gradient
    end
end


function run_batch!(launch!,repeats)
    for _ in 1:repeats
        launch!()
    end
    nothing
end
function study_main(prepare,args=ARGS)
    repeats=length(args)>0 ? parse(Int,args[1]) : 50;repeats>0||error("positive repeatsrequired")
    batch=2;C=8;H=128;W=128;OH=30;OW=30;n=batch*C*H*W;m=batch*C*OH*OW
    input=zeros(Float32,n);output=zeros(Float32,m);dy=fill(Float32(H*W),m);expected=zeros(Float32,n)
    study_reference!(n,dy,C,H,W,OH,OW,expected)
    din=CuArray(input);dout=CuArray(output);ddy=CuArray(dy);dx=CUDA.zeros(Float32,n)
    launch! = prepare(din,dout,ddy,dx,batch,C,H,W,OH,OW)
    run_batch!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.fill!(dx,0f0);CUDA.synchronize();gb=Base.gc_num()
        t0=time_ns();run_batch!(launch!,repeats);CUDA.synchronize();secs=(time_ns()-t0)*1e-9/repeats
        ga=Base.gc_num();gct=(ga.total_time-gb.total_time)*1e-9;gcp=ga.pause-gb.pause
        got=Array(dx);err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));ok=all(isfinite,got)&&ma<=1f-3;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"pool_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,string(ok),n,ma,mr,repeats,gct,gcp,file,n)
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
