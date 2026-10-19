using Printf
const P=@__DIR__
include("study_params.jl")
function read_values(name,T,n)
    p=joinpath(P,name);filesize(p)==n*sizeof(T)||error("Invalid input size $name")
    open(p) do f;read!(f,Vector{T}(undef,n));end
end
guarded(a,g)=vcat(fill(g,16),a,fill(g,16))
function make_work(k)
    w=WS[k];h=HS[k];kw=KWS[k];kh=KHS[k];n=NS[k]
    image=guarded(read_values("image$(k-1).bin",UInt8,3*Int(w)*Int(h)),UInt8(231))
    kernel=guarded(read_values("kernel$(k-1).bin",UInt8,3*Int(kw)*Int(kh)),UInt8(231))
    sad=guarded(fill(Int32(-777),Int(n)),Int32(-99999))
    minimum=guarded(Int32[20],Int32(-99999));count=guarded(Int32[0],Int32(-99999))
    host=(image,kernel,sad,minimum,count);device=map(allocate_model,host)
    (;w,h,kw,kh,n,host,device)
end
function reset!(w)
    for (a,b) in zip(w.device,w.host);copyto!(a,b);end
end
function prefix!(out,w)
    append!(out,Int32[w.w,w.h,w.kw,w.kh,w.n,w.kw*w.kh,3])
    for a in w.device;append!(out,Array(a));end
end
function pipeline(w)
    image,kernel,sad,minimum,count=w.device
    seconds=0.0
    for r=1:3
        # Source does these two initializations outside its three-kernel ROI.
        copyto!(minimum,17,Int32[20],1,1);copyto!(count,17,Int32[0],1,1)
        CUDA.device_synchronize();start=time_ns()
        launch_compute(sad,image,kernel,w.w,w.h,w.kw,w.kh)
        launch_minimum(sad,minimum,w.n)
        launch_count(sad,minimum,count,w.n)
        CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
    end
    seconds/3
end
function tail!(out,w)
    image,kernel,sad,minimum,count=w.device
    for a in (sad,minimum,count,image,kernel);append!(out,Array(a));end
end
function main()
    expected=read_values("oracle_full_i32.bin",Int32,COUNT);work=map(make_work,1:5)
    for w in work;pipeline(w);end
    for trial=1:5
        out=Int32[]
        for w in work;reset!(w);prefix!(out,w);end
        length(out)==PREFIX||error("prefix length")
        seconds=0.0
        for w in work;seconds+=pipeline(w);end
        for w in work;tail!(out,w);end
        length(out)==COUNT||error("capture length")
        good=out==expected;maxerr=maximum(abs.(Int64.(out).-Int64.(expected)))
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"sad_$(MODEL)_trial$(trial).bin")
        open(path,"w") do io;write(io,out);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"fivecase_source3GPUstages_R3regionmean_sum_contextsync\",\"iterations\":3,\"internal_repeats\":3,\"kernel_launches\":45,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,good ? "true" : "false",length(out),maxerr,path,length(out));flush(stdout)
        good||error("Full independent exact reference rejection")
    end
end
