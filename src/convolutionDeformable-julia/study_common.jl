read_f32(name)=copy(reinterpret(Float32,read(joinpath(@__DIR__,name))))
guarded(x)=vcat(fill(777.0f0,16),x,fill(777.0f0,16))
function make_work(s)
    C,G,H,W,KH,KW,PH,PW,SH,SW,DH,DW,OH,OW=Int32.((s.C,s.G,s.H,s.W,s.KH,s.KW,s.PH,s.PW,s.SH,s.SW,s.DH,s.DW,s.OH,s.OW))
    p=(;C,G,H,W,KH,KW,PH,PW,SH,SW,DH,DW,OH,OW)
    image=guarded(read_f32("case$(s.id)_image_f32.bin"));offset=guarded(read_f32("case$(s.id)_offset_f32.bin"));mask=guarded(read_f32("case$(s.id)_mask_f32.bin"));columns=guarded(read_f32("case$(s.id)_colgrad_f32.bin"))
    initial=(;image,offset,mask,columns,gi=guarded(zeros(Float32,C*H*W)),go=guarded(zeros(Float32,G*2*KH*KW*OH*OW)),gm=guarded(zeros(Float32,G*KH*KW*OH*OW)))
    (;s,p,initial,device=map(allocate_model,initial))
end
function reset_work!(w)
    for name in keys(w.initial);copyto!(getproperty(w.device,name),getproperty(w.initial,name));end
end
function prefix!(a,w)
    s=w.s;append!(a,Float64.((s.id,s.sample,s.C,s.G,s.H,s.W,s.KH,s.KW,s.PH,s.PW,s.SH,s.SW,s.DH,s.DW,s.OH,s.OW)))
    for name in (:image,:offset,:mask,:columns,:gi,:go,:gm);append!(a,Float64.(Array(getproperty(w.device,name))));end
end
function tail!(a,w)
    for name in (:columns,:gi,:go,:gm);append!(a,Float64.(Array(getproperty(w.device,name))));end
end
function check_output(a,e,works)
    good=length(a)==length(e);err=0.0
    for i in eachindex(e)
        error=abs(a[i]-e[i]);err=max(err,error);good&=isfinite(a[i])&&error<=(i<=PREFIX ? 0.0 : 2e-6+2e-6*abs(e[i]))
    end
    off=PREFIX
    for w in works,name in (:columns,:gi,:go,:gm)
        n=length(getproperty(w.initial,name))-32
        good&=all(==(777.0),view(a,off+1:off+16))&&all(==(777.0),view(a,off+n+17:off+n+32));off+=n+32
    end
    (good,err)
end
function study_main(model)
    expected=copy(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));works=map(make_work,CASES)
    for w in works;launch_pipeline!(w);CUDA.device_synchronize();end # actualall3GPUbody/allgeometry warmup outside five trials
    for trial=1:5
        foreach(reset_work!,works);out=Float64[];foreach(w->prefix!(out,w),works);length(out)==PREFIX||error("prefix extent")
        seconds=0.0
        for w in works
            CUDA.device_synchronize();t=time_ns();launch_pipeline!(w);CUDA.device_synchronize();seconds+=(time_ns()-t)/1e9
        end
        foreach(w->tail!(out,w),works);good,err=check_output(out,expected,works);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"convolutionDeformable_trial$(trial).bin");write(path,out)
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$seconds,\"correct\":$good,\"checked_values\":$(length(out)),\"max_abs_error\":$err,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum4batch_backward_coord_col2im_im2col_customkernel_triplets_completion\",\"internal_repeats\":1,\"iterations\":1,\"kernel_launches\":12,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float64\",\"count\":$(length(out)),\"byte_order\":\"little\"}}")
        flush(stdout);good||error("Independentfullshape-function/adjoint contract failure")
    end
end
