using Printf
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const PLAN_DIR=@__DIR__
function loadbytes(name)
    path=joinpath(PLAN_DIR,name); isfile(path)||error("missing fixture $path");read(path)
end
function guarded(x::Vector{UInt8})
    v=fill(UInt8(0xa5),length(x)+128);copyto!(v,65,x,1,length(x));v
end
function allocbuf(bytes::Vector{UInt8},::Type{T}) where T
    v=guarded(bytes); length(v)%sizeof(T)==0||error("alignment"); d=CuArray(collect(reinterpret(T,v)));d,view(d,div(64,sizeof(T))+1:length(d)-div(64,sizeof(T)))
end
bytes(d)=collect(reinterpret(UInt8,Array(d)))
function main()
    a=loadbytes("anchors_u16.bin");c=loadbytes("control_u8.bin");ro=loadbytes("oracle_return_i32.bin");mo=loadbytes("oracle_max_i32.bin");po=loadbytes("oracle_parent_i32.bin");length(a)==46835712||error("anchor extent");batches=div(length(c),8*1792)
    expectedpre=vcat(guarded(a),guarded(c));expectedout=vcat(guarded(ro),guarded(mo),guarded(po))
    for trial=0:5
        Ab,A=allocbuf(a,UInt64);Cb,C=allocbuf(c,UInt64);Rb,R=allocbuf(zeros(UInt8,length(ro)),Int32);Mb,M=allocbuf(zeros(UInt8,length(mo)),Int32);Pb,P=allocbuf(fill(UInt8(255),length(po)),Int32)
        actualpre=vcat(bytes(Ab),bytes(Cb));CUDA.device_synchronize()
        t=time_ns()
        for batch=0:batches-1
            launch!(view(R,batch*1792*1024*2+1:(batch+1)*1792*1024*2),view(A,batch*1792*1089+1:(batch+1)*1792*1089),view(C,batch*1792+1:(batch+1)*1792),M,P)
        end
        CUDA.device_synchronize();elapsed=(time_ns()-t)*1e-9
        actualout=vcat(bytes(Rb),bytes(Mb),bytes(Pb))
        if actualpre!=expectedpre || actualout!=expectedout || bytes(Ab)!=guarded(a) || bytes(Cb)!=guarded(c)
            ix=findfirst(actualout.!=expectedout);error("full byte mismatch index=$ix actual=$(ix===nothing ? -1 : actualout[ix]) expected=$(ix===nothing ? -1 : expectedout[ix])")
        end
        if trial>0
            path=joinpath(ENV["STUDY_OUTPUT_DIR"],"minimap2_trial$(trial).bin");open(path,"w") do f;write(f,actualpre);write(f,actualout);end
            @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"all_source_batched64lane_circular_DP_kernel_launches_and_wholedevice_completion_excludes_scheduler_allocation_capture\",\"iterations\":%d,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,length(actualpre)+length(actualout),batches,path,length(actualpre)+length(actualout));flush(stdout)
        end
    end
end
