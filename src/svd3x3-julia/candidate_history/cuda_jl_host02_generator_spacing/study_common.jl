using Printf
const PARENT=normpath(joinpath(@__DIR__))
const CHECK_LIB=joinpath(PARENT,"study_check.so")
const SVD_NS=(1,31,256,257,4097,65537)
const SVD_R=7
const SVD_PREFIX=2737581
const SVD_COUNT=2948346
function read_exact(name,::Type{T},n)where T
    b=read(joinpath(PARENT,name));length(b)==n*sizeof(T)||error("asset bytes $name")
    collect(reinterpret(T,b))
end
svd_pad(a)=vcat(fill(91.25f0,16),a,fill(91.25f0,16))
function run_svd(w)
    CUDA.device_synchronize(blocking=true)
    start=time_ns()
    for _ in 1:SVD_R;launch_svd(w);end
    CUDA.device_synchronize(blocking=true)
    (time_ns()-start)*1e-9/SVD_R
end
function scientific_check(a,o,sigma,n)
    scores=Vector{Float64}(undef,3*n);maxima=zeros(5)
    good=GC.@preserve a o sigma scores maxima begin
        ccall((:svd_check,CHECK_LIB),Cint,(Ptr{Cfloat},Ptr{Cfloat},Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble}),pointer(a,17),pointer(o,17),pointer(sigma),n,pointer(scores),pointer(maxima))
    end
    (good==1,scores,maxima)
end
function study_main(model)
    work=map(enumerate(SVD_NS))do (id,n)
        a=svd_pad(read_exact("case$(id-1)_input_f32.bin",Float32,9*n))
        initial=svd_pad(fill(-777f0,21*n));sigma=read_exact("case$(id-1)_sigma_f64.bin",Float64,3*n)
        (id=id-1,n=n,a=a,initial=initial,sigma=sigma,di=allocate_model(a),dout=allocate_model(initial))
    end
    for w in work;run_svd(w);end
    rawdir=ENV["STUDY_RAW_DIR"];mkpath(rawdir)
    for trial in 1:5
        before=[];initial_echo=[]
        for w in work
            copyto!(w.di,w.a);copyto!(w.dout,w.initial)
            a=Array(w.di);b=Array(w.dout);a==w.a&&b==w.initial||error("actual initial state")
            push!(before,a);push!(initial_echo,b)
        end
        seconds=sum(run_svd(w)for w in work)
        after=[Array(w.di)for w in work];outputs=[Array(w.dout)for w in work]
        artifact=Float64[];sizehint!(artifact,SVD_COUNT)
        for (k,w)in enumerate(work)
            append!(artifact,Float64[w.n,SVD_R,256,w.n÷256+1]);append!(artifact,before[k]);append!(artifact,initial_echo[k]);append!(artifact,after[k])
        end
        length(artifact)==SVD_PREFIX||error("prefix count")
        good=true;maxima=zeros(5)
        for (k,w)in enumerate(work)
            ok,scores,err=scientific_check(after[k],outputs[k],w.sigma,w.n)
            guards=all(==(91.25f0),outputs[k][1:16])&&all(==(91.25f0),outputs[k][end-15:end])
            ok=ok&&guards&&after[k]==w.a;good=good&&ok;maxima=max.(maxima,err)
            append!(artifact,scores);append!(artifact,outputs[k][1:16]);append!(artifact,outputs[k][end-15:end]);append!(artifact,fill(ok ? 1.0 : 0.0,6))
            base=joinpath(rawdir,"$(model)_pid$(getpid())_trial$(trial)_case$(w.id)")
            write(base*"_input_before.bin",before[k]);write(base*"_output_initial.bin",initial_echo[k]);write(base*"_input_after.bin",after[k]);write(base*"_output_final.bin",outputs[k])
        end
        length(artifact)==SVD_COUNT||error("artifact count")
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"svd_trial$(trial).bin");write(path,artifact)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum_sixcase_source_R7_regionmeans_contextcompletion\",\"internal_repeats\":7,\"kernel_launches\":42,\"numerical_agreement_claim\":false,\"raw_output_identity_claim\":false,\"property_maxima\":[%.17g,%.17g,%.17g,%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,good ? "true" : "false",length(artifact),maxima...,path,length(artifact))
        flush(stdout);good||error("full factorization scientific check")
    end
end
