const BRIDGE=ARGS[1]
readtyped(T,name)=copy(reinterpret(T,read(joinpath(@__DIR__,name))))
guard(x,v)=vcat(fill(eltype(x)(v),16),x,fill(eltype(x)(v),16))
function study_main(model)
    ccall((:study_state_bytes,BRIDGE),Cint,())==48||error("ABI")
    yy=readtyped(Int32,"y_i32.bin");nn=readtyped(Float32,"n_f32.bin")
    initial=(y=guard(yy,1234567),n=guard(nn,-999),theta=guard(zeros(Float32,N),-999),state=guard(zeros(UInt32,N*12),0xdeadbeef),flat=guard(zeros(Float32,BLOCKS),-999),logs=guard(zeros(Float32,BLOCKS),-999))
    device=map(allocate_model,initial)
    function reset!()
        for key in keys(initial);copyto!(getproperty(device,key),getproperty(initial,key));end
        ccall((:study_seed_states,BRIDGE),Cint,(CuPtr{Cvoid},Cint),pointer(device.state,17),N)==0||error("setup")
        ccall((:study_reset,BRIDGE),Cvoid,())
    end
    function append_array!(out,a)
        append!(out,Float64.(Array(a)))
    end
    function run_trajectory!()
        reset!();out=Float64[N,STEPS,256,BLOCKS,123,20,1]
        for key in keys(initial);append_array!(out,getproperty(device,key));end
        length(out)==PREFIX||error("prefix")
        a=Ref(20f0);b=Ref(1f0);total=0.0;maxerr=0.0;good=true
        for step=0:STEPS-1
            before=Array(device.state);olda=a[];oldb=b[];CUDA.device_synchronize()
            start=time_ns();launch_sample!(device,olda,oldb);launch_sum!(device);CUDA.device_synchronize();total+=(time_ns()-start)/1e9
            after=Array(device.state);ht=Array(device.theta);hf=Array(device.flat);hl=Array(device.logs);err=Ref(0.0)
            bad=GC.@preserve before after ht yy nn begin
                ccall((:study_check,BRIDGE),Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Float32},Ptr{Int32},Ptr{Float32},Cfloat,Cfloat,Cint,Ref{Cdouble}),pointer(before,17),pointer(after,17),pointer(ht,17),pointer(yy),pointer(nn),olda,oldb,N,err)
            end
            maxerr=max(maxerr,err[]);good&=bad==0;fs=0f0;ls=0f0
            for j=0:BLOCKS-1
                ft=zeros(Float32,256);lt=zeros(Float32,256)
                for k=0:255
                    id=j*256+k
                    if id<N;ft[k+1]=ht[17+id];lt[k+1]=log(ht[17+id]);end
                end
                stride=128
                while stride>0
                    for k=1:stride;ft[k]+=ft[k+stride];lt[k]+=lt[k+stride];end
                    stride÷=2
                end
                good&=abs(Float64(ft[1])-hf[17+j])<=2e-4&&abs(Float64(lt[1])-hl[17+j])<=2e-4
                fs+=hf[17+j];ls+=hl[17+j]
            end
            ccall((:study_feedback,BRIDGE),Cvoid,(Ref{Cfloat},Ref{Cfloat},Cint,Cfloat,Cfloat),a,b,N,fs,ls)
            good&=isfinite(a[])&&isfinite(b[])&&a[]>0&&b[]>0
            for key in keys(initial)
                original=getproperty(initial,key);actual=Array(getproperty(device,key))
                good&=actual[1:16]==original[1:16]&&actual[end-15:end]==original[end-15:end]
            end
            append!(out,Float64[step,olda,oldb,a[],b[],fs,ls,bad,err[]])
            for key in (:theta,:flat,:logs,:state);append_array!(out,getproperty(device,key));end
            if !good
                println(stderr,"oraclefailure step=$step bad=$bad error=$(err[])");break
            end
        end
        good&=length(out)==COUNT
        out,total,good,maxerr
    end
    run_trajectory!() # Full exacthost/device JITwarmup andchecks outside all five trialtimes.
    for trial=1:5
        out,total,good,err=run_trajectory!();path=joinpath(ENV["STUDY_OUTPUT_DIR"],"gibbs_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$total,\"correct\":$good,\"checked_values\":$(length(out)),\"max_abs_error\":$err,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum32conditional_sampling_reduction_GPUcompletion_excludes_hostfeedback_capture\",\"internal_repeats\":32,\"iterations\":32,\"kernel_launches\":64,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float64\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("FullconditionalGamma/stateprogress/reduction/feedbackcontract failed")
    end
end
