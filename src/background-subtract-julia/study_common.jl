using CUDA
using Printf
const STUDY_N=65536
function study_fixture()
    n=STUDY_N;bn=UInt8[UInt8(i%256) for i in 0:n-1];a=UInt8[UInt8((i÷256)%256) for i in 0:n-1];b=UInt8[UInt8((i*29+11)%256) for i in 0:n-1]
    h1=copy(a);h2=UInt8[isodd(i) ? a[i] : UInt8(255-a[i]) for i in 1:n]
    bn[1:256].=21;a[1:256].=141;b[1:256].=0;h1[1:256].=141;h2[1:256].=141
    bn,fill(UInt8(128),n),a,b,h1,h2
end
@inline study_u8(x)=UInt8(trunc(Int32,x))
function study_cpu!(img,h1,h2,tn,bn)
    for i in eachindex(img)
        if abs(Int(img[i])-Int(h1[i]))<=tn[i] && abs(Int(img[i])-Int(h2[i]))<=tn[i]
            bn[i]=study_u8(.92*Float64(bn[i])+.08*Float64(img[i]))
            th=Float32(.92*Float64(tn[i])+.24*Float64(Int(img[i])-Int(bn[i])))
            tn[i]=study_u8(max(th,20f0))
        end
    end
end
function study_pair!(launch!,mode,a,b,h1,h2,bn,tn,mp)
    launch!(mode,a,h1,h2,tn,bn,mp)
    launch!(mode,b,a,h1,tn,bn,mp)
end
function study_main(prepare,args=ARGS)
    repeats=isempty(args) ? 20 : parse(Int,args[1]);repeats>0||error("positive repeats")
    bn0,tn0,a,b,h1,h2=study_fixture();ebr=copy(bn0);etr=copy(tn0)
    study_cpu!(a,h1,h2,etr,ebr);study_cpu!(b,a,h1,etr,ebr);expected=vcat(ebr,etr,ebr,etr)
    da,db,dh1,dh2=CuArray(a),CuArray(b),CuArray(h1),CuArray(h2);bn,tn,mp=CuArray(bn0),CuArray(tn0),CUDA.zeros(UInt8,STUDY_N)
    launch! = prepare();study_pair!(launch!,0,da,db,dh1,dh2,bn,tn,mp);copyto!(bn,bn0);copyto!(tn,tn0);study_pair!(launch!,1,da,db,dh1,dh2,bn,tn,mp);CUDA.synchronize();allok=true
    for trial in 1:5
        times=Float64[];output=UInt8[];gct=0.0;gcp=0
        for mode in (0,1)
            total=0.0
            for rep in 1:repeats
                copyto!(bn,bn0);copyto!(tn,tn0);CUDA.synchronize();gb=Base.gc_num();t0=time_ns()
                study_pair!(launch!,mode,da,db,dh1,dh2,bn,tn,mp);CUDA.synchronize();total+=(time_ns()-t0)*1e-9
                ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
            end
            push!(times,total/repeats);append!(output,Array(bn));append!(output,Array(tn))
        end
        ma=maximum(abs.(Int.(output).-Int.(expected)));ok=ma==0;allok &=ok
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"background_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum_split_and_merged_two_frame_kernel_regions\",\"jit_excluded\":true,\"internal_repeats\":%d,\"mode_times_seconds\":[%.12g,%.12g],\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,repeats,times[1],times[2],gct,gcp,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
