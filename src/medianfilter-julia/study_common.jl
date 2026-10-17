using CUDA
using Printf
isdefined(@__MODULE__,:median_pixel)||include(joinpath(@__DIR__,"median_math.jl"))
function exact_rank_oracle(input,W,H)
    out=zeros(UInt8,3W*H)
    for y in 0:H-1,x in 0:W-1,ch in 0:2
        values=UInt8[]
        for dy in -1:1,dx in -1:1
            xx=x+dx;yy=y+dy;p=(0<=xx<W&&0<=yy<H) ? input[yy*W+xx+1] : UInt32(0);push!(values,UInt8((p>>UInt32(8ch))&0xff))
        end
        out[3*(y*W+x)+ch+1]=sort(values)[5]
    end
    out
end
function unpack_pixels(pixels)
    UInt8[UInt8((p>>UInt32(8ch))&0xff) for p in pixels for ch in 0:2]
end
function study_main(prepare,args=ARGS)
    W=160;H=128;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");input=UInt32[UInt32((x*13+y*7)%256)|(UInt32((x*x+3y)%256)<<8)|(UInt32((5x+y*y+17)%256)<<16) for y in 0:H-1 for x in 0:W-1];rank=exact_rank_oracle(input,W,H);di,dc=CuArray(input),CUDA.zeros(UInt32,W*H);launch! = prepare(di,dc,W,H);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds=(time_ns()-t0)*1e-9/repeat;pixels=Array(dc);got=unpack_pixels(pixels);rankerr=maximum(abs.(Int32.(got).-Int32.(rank)));ok=rankerr<=1&&all((pixels.>>24).==0);allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"medianfilter_trial$(trial).bin");write(file,pixels)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"true_rank_median_max_byte_error\":%d,\"timing_region\":\"full_8step_RGB_binary_median_zero_extended_3x3\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(pixels),rankerr,rankerr,repeat,file,length(pixels))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
