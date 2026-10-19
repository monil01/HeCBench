# Identical scientific/measurement host harness, outside each model's GPU body.
using Libdl
include(joinpath(@__DIR__,"host_specs.jl"))
readarray(T,path)=reinterpret(T,read(path)) |> collect
function study_main(launch,allocate)
    raw=readarray(UInt32,"common.bin")
    vals=ntuple(i->i==8 ? reinterpret(Float32,raw[i]) : reinterpret(Int32,raw[i]),97)
    common=NamedTuple{COMMON_NAMES}(vals)
    names=Tuple(s[1] for s in FIELD_SPECS)
    initial=map(FIELD_SPECS) do spec
        name,T,n,guard=spec
        a=zeros(T,n+32); a[1:16].=T(guard); a[end-15:end].=T(guard)
        path=string(name)*"_i32.bin"
        if name in (:endoRow,:endoCol,:epiRow,:epiCol)
            a[17:end-16].=readarray(T,path)
        end
        a
    end
    dev=map(allocate,initial)
    image0=zeros(Float32,Int(common.frame_elem)+32);image0[1:16].=-2047.5f0;image0[end-15:end].=-2047.5f0
    image=allocate(image0)
    buffers=merge(NamedTuple{names}(dev),(frame=image,))
    frames=[readarray(Float32,"frame$(i)_f32.bin") for i=0:FRAMES-1]
    expectedPrefix=readarray(Float32,"oracle_prefix_f32.bin")
    checker=Libdl.dlsym(Libdl.dlopen(joinpath(pwd(),"study_checker.so")),:check_heartwall)
    function check(x,frame,last)
        err=Ref{Cdouble}(0)
        good=ccall(checker,Cint,(Ptr{Float32},Cint,Cint,Cint,Ref{Cdouble}),x,length(x),frame,last,err)
        good==1 || error("Independent full field oracle rejected frame $frame; max mixed-unit error $(err[])")
        err[]
    end
    reset()=(foreach(pair->copyto!(pair[1],pair[2]),zip(dev,initial));copyto!(image,image0);nothing)
    snapshot()=vcat(map(a->Float32.(Array(a)),dev)...)
    baseprefix()=vcat(Float32[11,51,256,97,30],Float32[vals...],snapshot())
    setframe(frame)=(copyto!(image,17,frames[frame+1],1,length(frames[frame+1]));nothing)
    # Full pipeline + exact same specializations warmed before trial1, including CPU checking.
    reset()
    for frame=0:FRAMES-1
        setframe(frame);launch(Int32(frame),common,buffers);CUDA.device_synchronize(blocking=true)
    end
    check(snapshot(),10,true)
    for trial=1:5
        reset(); out=baseprefix(); seconds=0.0
        for frame=0:FRAMES-1
            setframe(frame);append!(out,Array(image));CUDA.device_synchronize(blocking=true)
            start=time_ns();launch(Int32(frame),common,buffers);CUDA.device_synchronize(blocking=true)
            seconds+=(time_ns()-start)*1e-9
        end
        final=snapshot(); maxerr=check(final,10,true)
        correct=length(out)==PREFIX && reinterpret(UInt32,out)==reinterpret(UInt32,expectedPrefix)
        append!(out,final);correct &= length(out)==COUNT
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"heartwall_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$correct,\"checked_values\":$(length(out)),\"max_abs_error\":$maxerr,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum11full_heartwall_GPU_37stage_kernel_completion_excludes_frameH2D_and_capture\",\"internal_repeats\":11,\"iterations\":11,\"kernel_launches\":11,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float32\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);correct || error("Input prefix mismatch")
    end
    # Separate replay, no claim to observing overwritten intermediates in the timed trial.
    reset();mkpath(ARGS[1]);path=joinpath(ARGS[1],"actual_all_frame_states_$(getpid()).bin")
    open(path,"w") do io
        write(io,baseprefix())
        for frame=0:FRAMES-1
            setframe(frame);one=vcat(Float32[frame],Array(image));launch(Int32(frame),common,buffers);CUDA.device_synchronize(blocking=true)
            state=snapshot();check(state,frame,frame==10);append!(one,state);write(io,one)
            println("STUDY_AUXILIARY actual_untimed_full_frame_state frame=$frame correct=true values=$(length(one)) path=$path");flush(stdout)
        end
    end
end
