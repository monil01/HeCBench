readtyped(T,name)=copy(reinterpret(T,read(joinpath(@__DIR__,name))))
guard(x,v)=vcat(fill(eltype(x)(v),16),x,fill(eltype(x)(v),16))
const PARAMS=(l=Int32(33),d=Int32(50),rp=.005,re=.05,decay=.9,firstgap=0.0,othergap=0.0,minmask=.5)
function make_work(case,n)
    initial=guard(readtyped(UInt8,"case$(case)_input_u8.bin"),211)
    (;case,n,initial,seq=allocate_model(initial))
end
function prefix!(out,mat,table,works)
    append!(out,Float64[33,50,.005,.05,.9,0,0,.5,128,3])
    append!(out,Float64.(Array(mat)));append!(out,Float64.(Array(table)))
    for w in works
        append!(out,Float64[w.case,w.n,(w.n+128)÷128]);append!(out,Float64.(Array(w.seq)))
    end
end
function study_main(model)
    expected=readtyped(Float64,"oracle_primary_f64.bin");posterior=readtyped(Float64,"oracle_posterior_f64.bin")
    mat=allocate_model(guard(readtyped(Float64,"likelihood_f64.bin"),-777))
    table=allocate_model(guard(readtyped(UInt8,"mask_table_u8.bin"),211))
    works=[make_work(i-1,NS[i]) for i=1:3]
    reset!()=foreach(w->copyto!(w.seq,w.initial),works)
    pipeline!()=begin
        for w in works;launch_mask!(w,mat,table,PARAMS);end
        CUDA.device_synchronize()
    end
    pipeline!() # Exact primary kernels warmed outside all timed trials.
    for trial=1:5
        reset!();out=Float64[];prefix!(out,mat,table,works);length(out)==PREFIX||error("prefix")
        CUDA.device_synchronize();start=time_ns();pipeline!();seconds=(time_ns()-start)/1e9
        for w in works;append!(out,Float64.(Array(w.seq)));end
        length(out)==COUNT||error("extent");good=all(isfinite,out)&&out==expected;err=maximum(abs.(out.-expected))
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"diamond_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$good,\"checked_values\":$(length(out)),\"max_abs_error\":$err,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"resident3batch_fixed33_Tantan_HMM_masking_kernel_completion\",\"internal_repeats\":1,\"iterations\":1,\"kernel_launches\":3,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float64\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("Independent51stateHMM exact mask contract")
    end
    reset!();trace=Float64[];prefix!(trace,mat,table,works);offset=0;maximum_error=0.0;good=true
    for w in works
        post=allocate_model(guard(zeros(Float32,w.n*33),-999));errors=allocate_model(guard(zeros(Int32,w.n),-12345))
        launch_observer!(w,mat,PARAMS,post,errors);CUDA.device_synchronize()
        hp=Array(post);he=Array(errors)
        for v in view(hp,17:16+w.n*33)
            offset+=1;e=abs(v-posterior[offset]);maximum_error=max(maximum_error,e)
            good&=isfinite(v)&&e<=3e-7+3e-7*abs(posterior[offset])
        end
        good&=all(==(0),view(he,17:16+w.n));good&=all(==(-999),hp[1:16])&&all(==(-999),hp[end-15:end]);good&=all(==(-12345),he[1:16])&&all(==(-12345),he[end-15:end])
        append!(trace,Float64.(hp));append!(trace,Float64.(he))
    end
    directory=ARGS[1];mkpath(directory);path=joinpath(directory,"actual_posterior_$(getpid()).bin");write(path,trace)
    println("STUDY_AUXILIARY actual_untimed_helper_posterior correct=$good checked_posterior_values=$offset total_artifact_values=$(length(trace)) max_abs_error=$maximum_error path=$path")
    flush(stdout);good||error("Independent posterior/guards contract")
end
