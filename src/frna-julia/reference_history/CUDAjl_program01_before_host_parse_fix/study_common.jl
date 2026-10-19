using Printf
const RNA_ROOT=@__DIR__
struct RNAGuarded{T}
    base::CuArray{T,1}
    payload::CuArray{T,1}
    initial::Vector{T}
    n::Int
end
function RNAGuarded(x::Vector{T}) where T
    guard=reinterpret(T,UInt32(0xa5a5a5a5))
    initial=vcat(fill(guard,32),x,fill(guard,32));base=CuArray(initial)
    payload=unsafe_wrap(CuArray,pointer(base,33),length(x);own=false)
    RNAGuarded(base,payload,initial,length(x))
end
rna_reset!(g::RNAGuarded)=(copyto!(g.base,g.initial);nothing)
function rna_take(g::RNAGuarded)
    x=Array(g.base);@assert x[1:32]==g.initial[1:32]&&x[end-31:end]==g.initial[end-31:end]
    x
end
function rna_read(path,::Type{T}) where T
    @assert isfile(path)&&filesize(path)%sizeof(T)==0
    open(path,"r")do f
        x=Vector{T}(undef,filesize(path)÷sizeof(T));read!(f,x);@assert eof(f);x
    end
end
struct RNACase
    n::Int32
    seq::RNAGuarded{Int32}
    v::RNAGuarded{Int32}
    w::RNAGuarded{Int32}
    wm::RNAGuarded{Int32}
    wca::RNAGuarded{Int32}
    w5::RNAGuarded{Int32}
    w3::RNAGuarded{Int32}
end
function RNACase(seq::Vector{Int32})
    n=length(seq)
    RNACase(Int32(n),RNAGuarded(seq),RNAGuarded(fill(RNA_INF,n*n)),RNAGuarded(fill(RNA_INF,n*n)),
        RNAGuarded(fill(RNA_INF,n*n)),RNAGuarded(fill(RNA_INF,n*n)),
        RNAGuarded(fill(Int32(-777777),n+1)),RNAGuarded(fill(Int32(-777777),n+1)))
end
rna_buffers(c::RNACase)=(c.v,c.w,c.wm,c.wca,c.w5,c.w3)
function rna_reset!(c::RNACase)
    foreach(rna_reset!,rna_buffers(c));nothing
end
rna_context(c,par)=(seq=c.seq.payload,v=c.v.payload,w=c.w.payload,wm=c.wm.payload,wca=c.wca.payload,w5=c.w5.payload,w3=c.w3.payload,params=par.payload)
function rna_check!(c,gold,offset,out)
    for g in rna_buffers(c)
        x=rna_take(g);append!(out,Float64.(x));got=@view x[33:end-32]
        @assert offset+g.n<=length(gold)
        expected=@view gold[offset+1:offset+g.n]
        if got!=expected
            mismatch=findfirst(got.!=expected)
            write("diagnostic_gpu_array_Int32.bin",x)
            error("FRNA_SOURCE_CPU_MISMATCH offset=$offset index=$mismatch actual=$(got[mismatch]) expected=$(expected[mismatch])")
        end
        offset+=g.n
    end
    @assert Array(c.seq.base)==c.seq.initial
    offset
end
function rna_study()
    params=rna_read(joinpath(RNA_ROOT,"params.bin"),UInt32);@assert length(params)*4==365836
    par=RNAGuarded(params);raw=rna_read(joinpath(RNA_ROOT,"sequences.bin"),Int32)
    @assert raw[1]==6;k=2;cases=RNACase[]
    for index in 1:6
        n=Int(raw[k]);k+=1;@assert 4<n<=77&&k+n-1<=length(raw)
        s=raw[k:k+n-1];k+=n;@assert all(0 .<=s .<4);push!(cases,RNACase(s))
    end
    @assert k==length(raw)+1
    gold=rna_read(joinpath(RNA_ROOT,"serial_source_crosscheck.bin"),Int32);@assert length(gold)==2205408
    contexts=[rna_context(c,par)for c in cases]
    for (c,context)in zip(cases,contexts)
        rna_reset!(c);launch_init(context,c.n)
        for d in Int32(0):c.n-Int32(2);launch_step(context,c.n,d);end
    end
    CUDA.device_synchronize()
    for trial in 1:5
        prefix=Float64.(rna_take(par));out=Float64[];times=Float64[];mfes=Int32[];offset=0
        for c in cases
            rna_reset!(c);append!(prefix,Float64.(rna_take(c.seq)))
            for g in rna_buffers(c);append!(prefix,Float64.(rna_take(g)));end
        end
        @assert length(prefix)==128075
        if trial==1;write("diagnostic_actual_inputs_Float64.bin",prefix);end
        for (c,context)in zip(cases,contexts)
            CUDA.device_synchronize();t=time_ns();launch_init(context,c.n);CUDA.device_synchronize()
            push!(times,(time_ns()-t)*1e-9);offset=rna_check!(c,gold,offset,out)
            for d in Int32(0):c.n-Int32(2)
                CUDA.device_synchronize();t=time_ns();launch_step(context,c.n,d);CUDA.device_synchronize()
                push!(times,(time_ns()-t)*1e-9);offset=rna_check!(c,gold,offset,out)
            end
            a=rna_take(c.w5);b=rna_take(c.w3)
            @assert a[33]==0&&b[33+Int(c.n)]==0&&a[33+Int(c.n)]==b[33]&&b[33]<=0
            push!(mfes,b[33])
        end
        @assert offset==length(gold)&&mfes==Int32[-2760,0,-120,-40,0,-50]&&length(times)==164
        @assert Array(par.base)==par.initial
        append!(prefix,out);@assert length(prefix)==2396459
        name="output_trial$trial.bin";write(name,prefix)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":null,\"metric_scope\":\"ExactInt32fullsourceCPUstatecrosscheck; independentglobalMFEcertificateforfivecontrols, notfull77basebiologicaloptimum\",\"timing_region\":\"sum_initialization_and_perdistance_fullninekernel_contextcompleted_segments_excludes_captures\",\"jit_excluded\":true,\"internal_repeats\":1,\"input_prefix_count\":128075,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"MFE_scaled_Int32\":[%s],\"segment_seconds\":[%s]}\n",trial,sum(times),offset,name,length(prefix),join(mfes,","),join(times,","));flush(stdout)
    end
end
