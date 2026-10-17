using CUDA
using Printf
function study_reference!(lx::Int, ly::Int, threshold::Int, max_rad::Int, img, box, norm, out)
    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            sum = 0.0f0
            q = 1
            s = 1
            ksum = 0

            while sum < threshold && q < max_rad
                s = q
                sum = 0.0f0
                ksum = 0

                for i in -s:s
                    for j in -s:s
                        if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                            sum += img[(x + i) * ly + y + j + 1]
                            ksum += 1
                        end
                    end
                end
                q += 1
            end

            box[x * ly + y + 1] = s
            if ksum != 0
                inc = 1.0f0 / Float32(ksum)
                for i in -s:s
                    for j in -s:s
                        if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                            norm[(x + i) * ly + y + j + 1] += inc
                        end
                    end
                end
            end
        end
    end

    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            idx = x * ly + y + 1
            if norm[idx] != 0.0f0
                img[idx] /= norm[idx]
            end
        end
    end

    for x in 0:(lx - 1)
        for y in 0:(ly - 1)
            s = box[x * ly + y + 1]
            sum = 0.0f0
            ksum = 0
            for i in -s:s
                for j in -s:s
                    if x - s >= 0 && x + s < lx && y - s >= 0 && y + s < ly
                        sum += img[(x + i) * ly + y + j + 1]
                        ksum += 1
                    end
                end
            end
            if ksum != 0
                out[x * ly + y + 1] = sum / Float32(ksum)
            end
        end
    end
end


function study_main(prepare,args=ARGS)
    w=128;n=w*w;threshold=4000;radius=5;repeat=isempty(args) ? 10 : parse(Int,args[1]);repeat>0||error("positive repeat");ccall(:srand,Cvoid,(Cuint,),Cuint(123));input=Float32[Float32(ccall(:rand,Cint,())%256) for _ in 1:n]
    eimg=copy(input);ebox=zeros(Int32,n);enorm=zeros(Float32,n);eout=zeros(Float32,n);study_reference!(w,w,threshold,radius,eimg,ebox,enorm,eout);expected=vcat(eimg,enorm,Float32.(ebox),eout)
    img=CuArray(input);box=CUDA.zeros(Int32,n);norm=CUDA.zeros(Float32,n);out=CUDA.zeros(Float32,n);launch! = prepare(img,box,norm,out,w,threshold,radius);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        elapsed=0.0;gct=0.0;gcp=0
        for rep in 1:repeat
            copyto!(img,input);CUDA.fill!(box,Int32(0));CUDA.fill!(norm,0f0);CUDA.fill!(out,0f0);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();launch!();CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
        end
        got=vcat(Array(img),Array(norm),Float32.(Array(box)),Array(out));err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float32)));boxok=Array(box)==ebox;ok=all(isfinite,got)&&ma<=1f-3&&boxok;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"asmooth_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"box_exact\":%s,\"timing_region\":\"adaptive_box_norm_normalize_and_output_three_kernels_resets_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,mr,string(boxok),repeat,gct,gcp,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
