readtyped(T,name)=copy(reinterpret(T,read(joinpath(@__DIR__,name))))
guard(x,v)=vcat(fill(eltype(x)(v),16),x,fill(eltype(x)(v),16))
function study_main(model)
    expected=readtyped(Int32,"oracle_i32.bin");graphs=readtyped(Int32,"oracle_graph_i32.bin")
    regex=[allocate_model(guard(readtyped(UInt8,"regex$(j)_u8.bin"),211))for j=0:9]
    rt=allocate_model(guard(UInt32[0],1234567));match=allocate_model(Int32[256])
    # Matchcodeinput guarded inframeworkallocation; nativeStateonlyreads c=256.
    match=allocate_model(guard(Int32[256],1234567))
    works=[(id=k,n=(k==0 ? 1 : k==1 ? 4770 : 81923),bytes=(k==0 ? 6 : k==1 ? 144355 : 424072),text=allocate_model(guard(readtyped(UInt8,"text$(k)_u8.bin"),211)),offset=allocate_model(guard(readtyped(UInt32,"offset$(k)_u32.bin"),1234567)),result=[allocate_model(guard(zeros(UInt8,k==0 ? 1 : k==1 ? 4770 : 81923),211))for _=1:10])for k=0:2]
    pipeline!()=begin
        for w in works,j=1:10;launch_match!(w,regex[j],rt,match,w.result[j]);end
        CUDA.device_synchronize()
    end
    pipeline!() # Actual30primarydeviceentries/shapes +hostlaunch JIT warmup outside alltimings.
    for trial=1:5
        for w in works,out in w.result;copyto!(out,guard(zeros(UInt8,w.n),211));end
        out=Int32[3,10,512,160,100,1000,8000,Array(match)[17]];append!(out,Int32.(Array(rt)))
        for j=1:10
            append!(out,Int32[j-1,length(regex[j])-32]);append!(out,Int32.(Array(regex[j])))
        end
        for w in works
            append!(out,Int32[w.id,w.n,w.bytes]);append!(out,Int32.(Array(w.text)));append!(out,Int32.(Array(w.offset)))
            for a in w.result;append!(out,Int32.(Array(a)));end
        end
        length(out)==PREFIX||error("prefix")
        CUDA.device_synchronize();start=time_ns();pipeline!();seconds=(time_ns()-start)/1e9
        for w in works,a in w.result;append!(out,Int32.(Array(a)));end
        good=out==expected;path=joinpath(ENV["STUDY_OUTPUT_DIR"],"grep_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$good,\"checked_values\":$(length(out)),\"max_abs_error\":$(good ? \"0\" : \"null\"),\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"30resident_full_GPU_regex_postfix_NFAcompile_and_fullmatch_completion\",\"internal_repeats\":1,\"iterations\":1,\"kernel_launches\":30,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"int32\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("IndependentPythonregex fullmask/inputs/guards contract")
    end
    directory=ARGS[1];mkpath(directory)
    for j=1:10
        out=allocate_model(guard(fill(Int32(-999),512*404),-12345));launch_observer!(regex[j],rt,match,out);CUDA.device_synchronize();h=Array(out);ref=graphs[(j-1)*206880+1:j*206880];good=h==ref
        path=joinpath(directory,"actual_graph_$(j-1)_$(getpid()).bin");write(path,h)
        println("STUDY_AUXILIARY actual_untimed_NFA_graph pattern=$(j-1) correct=$good values=$(length(h)) path=$path");flush(stdout);good||error("ActualNFAgraph/source-semanticCPUcontrol")
    end
end
