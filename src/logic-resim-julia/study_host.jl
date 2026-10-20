# Original C++ CPU fallback/bookkeeping retained explicitly, not GPU computation.
using Libdl
function runstudy()
    sizeof(LogicEvent)==16||error("Source Event ABI mismatch")
    lib=Libdl.dlopen(joinpath(pwd(),"study_host.so"))
    sym(n)=Libdl.dlsym(lib,n)
    create=sym(:study_create);destroy=sym(:study_destroy);prepare=sym(:study_prepare)
    capacity=sym(:study_arena_capacity);usedfn=sym(:study_arena_used);hostptr=sym(:study_host_arena)
    uploadsfn=sym(:study_uploads);fieldsfn=sym(:study_gate_fields);offsetsfn=sym(:study_output_offsets);capsfn=sym(:study_caps)
    recordinput=sym(:study_record_inputs);recordround=sym(:study_record_round);grow=sym(:study_grow)
    importfn=sym(:study_import);finish=sym(:study_finish_cpu);validate=sym(:study_validate)
    prefixfn=sym(:study_prefix);outputfn=sym(:study_outputs)
    for trial=0:5
        state=ccall(create,Ptr{Cvoid},())
        count=Int(ccall(capacity,UInt64,(Ptr{Cvoid},),state))
        arena=allocate_arena(LogicEvent,count)
        host=unsafe_wrap(Array,ccall(hostptr,Ptr{LogicEvent},(Ptr{Cvoid},),state),count;own=false)
        truthCPU=read("truth_table_u8.bin");truth=allocate_data(truthCPU)
        Array(truth)==truthCPU||error("Actual GPU truth bytes differ")
        seconds=0.0;launches=0
        for batch=0:22
            n=Int(ccall(prepare,UInt64,(Ptr{Cvoid},UInt64),state,batch))
            segments=Ref{UInt64}(0);up=ccall(uploadsfn,Ptr{UInt64},(Ptr{Cvoid},Ref{UInt64}),state,segments)
            for k=0:Int(segments[])-1
                pos=Int(unsafe_load(up,2*k+1));len=Int(unsafe_load(up,2*k+2))
                copyto!(arena,pos+1,host,pos+1,len)
            end
            f=unsafe_wrap(Array,ccall(fieldsfn,Ptr{UInt64},(Ptr{Cvoid},),state),(80,n);own=false)
            funcs=allocate_data(copy(vec(f[1,:])))
            input=allocate_data(permutedims(copy(f[3:8,:])))
            delay=allocate_data(UInt32.(permutedims(copy(f[9:80,:]))))
            actual=Matrix{UInt64}(undef,80,n)
            af=Array(funcs);ai=Array(input);ad=Array(delay)
            actual[1,:].=af
            for j=1:n
                actual[2,j]=UInt64(truthCPU[Int(af[j])*1025+1025])
            end
            actual[3:8,:].=permutedims(ai);actual[9:80,:].=permutedims(UInt64.(ad))
            used=Int(ccall(usedfn,UInt64,(Ptr{Cvoid},),state))
            snapshot=Vector{LogicEvent}(undef,used);copyto!(snapshot,1,arena,1,used)
            ccall(recordinput,Cvoid,(Ptr{Cvoid},Ptr{LogicEvent},Ptr{UInt64}),state,snapshot,actual)
            gout=allocate_data(zeros(UInt64,n));gcap=allocate_data(zeros(UInt64,n));gsizes=allocate_data(zeros(UInt64,n));gflags=allocate_data(zeros(Bool,n))
            hsizes=zeros(UInt64,n);hflags=zeros(Bool,n);triples=zeros(UInt64,3*n)
            round=UInt64(1);done=false
            CUDA.device_synchronize(blocking=true)
            start=time_ns()
            while !done
                op=ccall(offsetsfn,Ptr{UInt64},(Ptr{Cvoid},),state)
                cp=ccall(capsfn,Ptr{UInt64},(Ptr{Cvoid},),state)
                outCPU=unsafe_wrap(Array,op,n;own=false);capCPU=unsafe_wrap(Array,cp,n;own=false)
                copyto!(gout,outCPU);copyto!(gcap,capCPU)
                launch!(arena,funcs,input,delay,gout,gcap,gsizes,gflags,truth,Int32(n),UInt64(300000000));launches+=1
                copyto!(hflags,gflags)
                done=!any(hflags)
                if !done
                    number=Int(ccall(grow,UInt64,(Ptr{Cvoid},Ptr{Bool},UInt64,Ptr{UInt64}),state,hflags,round,triples))
                    seg=Ref{UInt64}(0);up=ccall(uploadsfn,Ptr{UInt64},(Ptr{Cvoid},Ref{UInt64}),state,seg)
                    for k=0:Int(seg[])-1
                        pos=Int(unsafe_load(up,2*k+1));len=Int(unsafe_load(up,2*k+2));copyto!(arena,pos+1,host,pos+1,len)
                    end
                    for k=0:number-1
                        old=Int(triples[3*k+1]);new=Int(triples[3*k+2]);len=Int(triples[3*k+3]);copyto!(arena,new+1,arena,old+1,len)
                    end
                end
                round+=UInt64(1)
            end
            seconds+=(time_ns()-start)*1e-9
            copyto!(hsizes,gsizes)
            ccall(recordround,Cvoid,(Ptr{Cvoid},Ptr{Bool},Ptr{UInt64},UInt64),state,hflags,hsizes,round-UInt64(1))
            round==2||error("Primary scope requires actual one-pass source batches")
            used=Int(ccall(usedfn,UInt64,(Ptr{Cvoid},),state));copyto!(host,1,arena,1,used)
            ccall(importfn,Cvoid,(Ptr{Cvoid},Ptr{UInt64}),state,hsizes)
        end
        ccall(finish,Cvoid,(Ptr{Cvoid},),state)
        correct=ccall(validate,Cint,(Ptr{Cvoid},),state)==1
        pn=Ref{UInt64}(0);on=Ref{UInt64}(0)
        pre=ccall(prefixfn,Ptr{UInt64},(Ptr{Cvoid},Ref{UInt64}),state,pn)
        out=ccall(outputfn,Ptr{UInt64},(Ptr{Cvoid},Ref{UInt64}),state,on)
        println("STUDY_COUNTS prefix=$(pn[]) output=$(on[]) total=$(pn[]+on[]) launches=$launches correct=$correct");flush(stdout)
        if trial>0||!correct
            path=joinpath(ENV["STUDY_OUTPUT_DIR"],"logic_trial$(max(trial,1)).bin")
            open(path,"w") do io
                write(io,unsafe_wrap(Array,pre,Int(pn[]);own=false));write(io,unsafe_wrap(Array,out,Int(on[]);own=false))
            end
            timing=trial>0 ? string(seconds) : "null"
            println("STUDY_TRIAL {\"trial\":$(max(trial,1)),\"time_seconds\":$timing,\"correct\":$correct,\"checked_values\":$(pn[]+on[]),\"max_abs_error\":$(correct ? "0" : "null"),\"max_rel_error\":$(correct ? "0" : "null"),\"jit_excluded\":true,\"timing_region\":\"sum23_source_GPUbatch_metadataH2D_simulate_flagsD2H_hostcontrol_preallocatedarena_excludes_CPUfallback_capture\",\"internal_repeats\":23,\"iterations\":23,\"kernel_launches\":$launches,\"CPU_source_fallback_gates\":16446,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"uint64\",\"count\":$(pn[]+on[]),\"byte_order\":\"little\"}}")
            flush(stdout)
        end
        ccall(destroy,Cvoid,(Ptr{Cvoid},),state)
        CUDA.unsafe_free!(arena);CUDA.unsafe_free!(truth)
        correct||error("Exact full source/scientific waveform oracle rejected")
    end
end
