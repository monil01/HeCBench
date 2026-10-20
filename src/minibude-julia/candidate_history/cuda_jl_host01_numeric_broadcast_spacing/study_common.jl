using Printf
const PARENT=normpath(@__DIR__)
const CASES=((65536,938,26,34),(1,938,26,34),(257,938,26,34),(31,1,1,34),(257,8,5,513),(257,8,5,513))
const REPEATS=7
const PREFIX=902221
const COUNT=968758
const BUFFER_NAMES=("protein","ligand","forcefield","pose0","pose1","pose2","pose3","pose4","pose5","energy")
function read_exact(name,::Type{T},n) where T
    b=read(joinpath(PARENT,name));length(b)==n*sizeof(T)||error("asset bytes $name")
    collect(reinterpret(T,b))
end
function pad(a::Vector{T}) where T
    words=fill(UInt32(0xa5a5a5a5),(length(a)+32)*(sizeof(T)÷4))
    out=collect(reinterpret(T,words));copyto!(out,17,a,1,length(a));out
end
raw_words(a)=collect(reinterpret(UInt32,a))
append_words!(out,a)=append!(out,Float64.(reinterpret(UInt32,a)))
device_buffers(w)=(w.protein,w.ligand,w.ff,w.poses...,w.out)
function run_bude(w)
    CUDA.device_synchronize(blocking=true)
    start=time_ns()
    for _ in 1:REPEATS;launch_bude(w);end
    CUDA.device_synchronize(blocking=true)
    (time_ns()-start)*1e-9/REPEATS
end
function study_main(model)
    work=map(enumerate(CASES)) do (id,shape)
        n,np,nl,nt=shape
        protein=pad(read_exact("case$(id-1)_protein.bin",BudeAtom,np))
        ligand=pad(read_exact("case$(id-1)_ligand.bin",BudeAtom,nl))
        ff=pad(read_exact("case$(id-1)_forcefield.bin",BudeFF,nt))
        allposes=read_exact("case$(id-1)_poses.bin",Float32,6*n)
        poses=ntuple(a->pad(allposes[(a-1)*n+1:a*n]),6)
        initial=pad(fill(-777f0,n));oracle=read_exact("case$(id-1)_oracle_channels_f64.bin",Float64,6*n)[1:6:6*n]
        host=(protein,ligand,ff,poses...,initial)
        (id=id-1,n=n,np=np,nl=nl,nt=nt,host=host,oracle=oracle,protein=allocate_model(protein),ligand=allocate_model(ligand),ff=allocate_model(ff),poses=map(allocate_model,poses),out=allocate_model(initial))
    end
    for w in work;run_bude(w);end
    rawdir=ENV["STUDY_RAW_DIR"];mkpath(rawdir)
    for trial in 1:5
        before=[]
        for w in work
            bufs=device_buffers(w)
            for (d,h) in zip(bufs,w.host);copyto!(d,h);end
            b=map(Array,bufs)
            all(raw_words(x)==raw_words(y) for (x,y) in zip(b,w.host))||error("actual initial inputs/outputs")
            push!(before,b)
        end
        component_seconds=[run_bude(w) for w in work]
        seconds=sum(component_seconds)
        after=[map(Array,device_buffers(w)) for w in work]
        record=Float64[];sizehint!(record,COUNT)
        for (k,w) in enumerate(work)
            append!(record,Float64[w.id,w.n,w.np,w.nl,w.nt,REPEATS,256,cld(w.n,256),1])
            for b in before[k];append_words!(record,b);end
            for b in after[k][1:9];append_words!(record,b);end
        end
        length(record)==PREFIX||error("prefix count")
        good=true;maxabs=0.;maxbudget=0.
        rawbase=joinpath(rawdir,"$(model)_pid$(getpid())_trial$(trial)")
        for (k,w) in enumerate(work)
            ok=all(raw_words(after[k][j])==raw_words(w.host[j]) for j in 1:9)
            output=after[k][10];words=raw_words(output)
            ok=ok&&all(==(UInt32(0xa5a5a5a5)),words[1:16])&&all(==(UInt32(0xa5a5a5a5)),words[end-15:end])
            energies=Float64.(output[17:w.n+16]);errors=abs.(energies.-w.oracle);budgets=.05.+.0002.*abs.(w.oracle)
            ok=ok&&all(isfinite,energies)&&all(errors.<=budgets);good=good&&ok
            maxabs=max(maxabs,maximum(errors));maxbudget=max(maxbudget,maximum(errors./budgets))
            append!(record,energies);append!(record,Float64.(words[1:16]));append!(record,Float64.(words[end-15:end]));push!(record,ok ? 1.0 : 0.0)
            for j in 1:10
                base=rawbase*"_case$(w.id)_$(BUFFER_NAMES[j])"
                write(base*"_before.bin",before[k][j]);write(base*"_after.bin",after[k][j])
            end
        end
        length(record)==COUNT||error("artifact count")
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"minibude_trial$(trial).bin");write(path,record)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"maximum_independent_error_over_budget\":%.17g,\"jit_excluded\":true,\"internal_repeats\":7,\"kernel_launches\":42,\"timing_region\":\"sixcase_sourceR7_regionmean_sum_completedwarmup_contextsync\",\"raw_state_prefix\":\"%s\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,good ? "true" : "false",length(record),maxabs,maxbudget,rawbase,path,length(record))
        println("STUDY_COMPONENT_TIMES trial=",trial," seconds=",repr(component_seconds)," scope=target_only_native_aggregate_only_no_crossmodel_size_claim")
        flush(stdout);good||error("full independent empirical energy/input/guard check")
    end
end
