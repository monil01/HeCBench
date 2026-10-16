using CUDA
using Printf
function study_reference!(input, output, w::Int, h::Int, a_square::Float32,
                    variance_I::Float32, variance_spatial::Float32, radius::Int)
    for idx0 in 0:w-1, idy0 in 0:h-1
        id = idy0 * w + idx0 + 1
        center = input[id]
        res = Float32(0)
        normalization = Float32(0)
        for oi in -radius:radius, oj in -radius:radius
            idk = idx0 + oi
            idl = idy0 + oj
            if idk < 0
                idk = -idk
            end
            if idl < 0
                idl = -idl
            end
            if idk > w - 1
                idk = w - 1 - oi
            end
            if idl > h - 1
                idl = h - 1 - oj
            end
            id_w = idl * w + idk + 1
            I_w = input[id_w]
            range = -((center - I_w) * (center - I_w)) / (Float32(2) * variance_I)
            spatial = -(Float32((idk - idx0)^2 + (idl - idy0)^2)) / (Float32(2) * variance_spatial)
            weight = a_square * exp(spatial + range)
            normalization += weight
            res += I_w * weight
        end
        output[id] = res / normalization
    end
    return output
end


function run_radius!(launch!,radius,repeats)
    for _ in 1:repeats
        launch!(radius)
    end
    nothing
end
function study_main(prepare,args=ARGS)
    W=length(args)>=1 ? parse(Int,args[1]) : 256;H=length(args)>=2 ? parse(Int,args[2]) : 256
    vi=length(args)>=3 ? parse(Float32,args[3]) : 100f0;vs=length(args)>=4 ? parse(Float32,args[4]) : 25f0;repeats=length(args)>=5 ? parse(Int,args[5]) : 5
    W>18&&H>18&&vi>0&&vs>0&&repeats>0||error("valid dimensions>18,variances,repeatsrequired")
    n=W*H;ccall(:srand,Cvoid,(Cuint,),Cuint(123));input=Float32[Float32(ccall(:rand,Cint,())%256) for _ in 1:n];a=.5f0/(vi*Float32(pi))
    radii=(Int32(3),Int32(6),Int32(9));expected=Vector{Vector{Float32}}()
    for r in radii
        e=zeros(Float32,n);study_reference!(input,e,W,H,a,vi,vs,Int(r));push!(expected,e)
    end
    din=CuArray(input);dout=CUDA.zeros(Float32,n);launch! = prepare(din,dout,W,H,a,vi,vs)
    for r in radii
        run_radius!(launch!,r,1)
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float32[];total=0.0;ma=0f0;mr=0f0;ok=true;gct=0.0;gcp=0;stage_times=Float64[]
        for (case,r) in enumerate(radii)
            CUDA.fill!(dout,0f0);CUDA.synchronize();gb=Base.gc_num()
            t0=time_ns();run_radius!(launch!,r,repeats);CUDA.synchronize();stage=(time_ns()-t0)*1e-9/repeats;total+=stage;push!(stage_times,stage)
            ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
            got=Array(dout);err=abs.(got.-expected[case]);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(expected[case]),eps(Float32))));ok &=all(isfinite,got)&&maximum(err)<=1f-3;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bilateral_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_radius3_6_9_kernel_launch_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"radius_times_seconds\":[%.12g,%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,total,string(ok),length(output),ma,mr,repeats,stage_times[1],stage_times[2],stage_times[3],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
