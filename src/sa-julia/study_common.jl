# Thrust C ABI preserves source container allocation/copy/sort/scan semantics.
# All custom source GPU math is authored in Julia and launched by each framework.
const SA_LIB=joinpath(@__DIR__,"libsa_thrust_bridge.so")
const SA_GUARD=Int32(0x13579bdf)
function sa_check(x)
    x==0 || error(unsafe_string(ccall((:sa_last_error,SA_LIB),Cstring,())))
    nothing
end
function sa_new(n)
    h=ccall((:sa_vec_new,SA_LIB),Ptr{Cvoid},(Csize_t,),n)
    h!=C_NULL||error("Thrustallocationfailed");h
end
sa_delete(h)=ccall((:sa_vec_delete,SA_LIB),Cvoid,(Ptr{Cvoid},),h)
function sa_view(h)
    n=ccall((:sa_vec_size,SA_LIB),Csize_t,(Ptr{Cvoid},),h)
    p=ccall((:sa_vec_data,SA_LIB),Ptr{Int32},(Ptr{Cvoid},),h)
    unsafe_wrap(CuArray,CuPtr{Int32}(UInt(p)),(Int(n),);own=false)
end
sa_assign(h,a)=sa_check(ccall((:sa_vec_assign_host,SA_LIB),Cint,(Ptr{Cvoid},Ptr{Int32},Csize_t),h,a,length(a)))
sa_copy(d,s)=sa_check(ccall((:sa_vec_assign_vec,SA_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid}),d,s))
function sa_read!(a,h)
    sa_check(ccall((:sa_vec_copy_host,SA_LIB),Cint,(Ptr{Cvoid},Ptr{Int32},Csize_t),h,a,length(a)));a
end
sa_read(h,n)=sa_read!(Vector{Int32}(undef,n),h)
function sa_scalar(h,i)
    x=Ref{Int32}();sa_check(ccall((:sa_vec_get,SA_LIB),Cint,(Ptr{Cvoid},Csize_t,Ref{Int32}),h,i,x));x[]
end
sa_sort(k,v,n)=sa_check(ccall((:sa_sort_by_key,SA_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t),k,v,n))
sa_scan(h,n)=sa_check(ccall((:sa_exclusive_scan,SA_LIB),Cint,(Ptr{Cvoid},Csize_t),h,n))
function sa_dc3(s_h,sa_h,n::Int32,K::Int32)
    n0=(n+Int32(2))÷Int32(3);n2=n÷Int32(3);n02=n0+n2
    h12=sa_new(n02+3);ha12=sa_new(n02+3);h0=sa_new(n0);ha0=sa_new(n0);hs=sa_new(n02+3)
    try
        s12=sa_view(h12);a12=sa_view(ha12);s0=sa_view(h0);a0=sa_view(ha0);scan=sa_view(hs);s=sa_view(s_h);sa=sa_view(sa_h)
        sa_launch(Val(:init),n02,s12,n02)
        for i in Int32(2):-Int32(1):Int32(0)
            sa_launch(Val(:key),n02,a12,s12,s,n02,i);sa_sort(ha12,h12,n02)
        end
        sa_copy(ha12,h12)
        sa_launch(Val(:scan),n02,s,a12,scan,n02);sa_scan(hs,n02+1)
        sa_launch(Val(:rank),n02,s12,a12,scan,n02,n0)
        maxrank=sa_scalar(hs,n02)
        if maxrank<n02
            sa_dc3(h12,ha12,n02,maxrank)
            sa_launch(Val(:store),n02,s12,a12,n02)
        else
            sa_launch(Val(:unique),n02,s12,a12,n02)
        end
        sa_launch(Val(:mod0),n02,a12,scan,n0,n02);sa_scan(hs,n02)
        sa_launch(Val(:set0),n02,s0,a12,scan,n0,n02)
        sa_launch(Val(:key),n0,a0,s0,s,n0,Int32(0));sa_sort(ha0,h0,n0);sa_copy(ha0,h0)
        sa_launch(Val(:merge),n0+n02,a0,a12,sa,s,s12,n0,n02,n)
    finally
        sa_delete(hs);sa_delete(ha0);sa_delete(h0);sa_delete(ha12);sa_delete(h12)
    end
    nothing
end
function sa_certify(inp,out,n)
    length(out)==n+20||return false
    all(==(Int32(0)),@view(out[n+1:n+3]))||return false
    all(==(SA_GUARD),@view(out[n+4:end]))||return false
    rank=fill(Int32(-2),n+1);rank[n+1]=Int32(-1)
    for i=1:n
        p=out[i];(0<=p<n && rank[p+1]==-2)||return false;rank[p+1]=Int32(i-1)
    end
    for i=1:n-1
        x=out[i];y=out[i+1]
        (inp[x+1]<inp[y+1]||(inp[x+1]==inp[y+1]&&rank[x+2]<rank[y+2]))||return false
    end
    true
end
function sa_case(c,repeats)
    n,_,inp=c;h=zeros(Int32,n+20);h[n+4:end].=SA_GUARD;s=sa_new(0);a=sa_new(0)
    try
        CUDA.device_synchronize(blocking=true);t=time_ns()
        for it=1:repeats
            sa_assign(s,inp);sa_assign(a,h);sa_dc3(s,a,Int32(n),Int32(26));sa_read!(h,a)
        end
        CUDA.device_synchronize(blocking=true);elapsed=(time_ns()-t)*1e-9/repeats
        actual=sa_read(s,n+20)
        return actual,h,elapsed,actual==inp&&sa_certify(inp,h,n)
    finally
        sa_delete(a);sa_delete(s)
    end
end
function sa_study()
    CUDA.stream!(CUDA.legacy_stream())
    which=ARGS[1];cases=[]
    for line in eachline(which*"_cases.txt")
        words=split(line);n=parse(Int,words[1]);r=parse(Int,words[2]);v=reinterpret(Int32,read(words[3]))|>collect
        @assert length(v)==n+20&&all(==(Int32(0)),v[n+1:n+3])&&all(==(SA_GUARD),v[n+4:end])
        push!(cases,(n,r,v))
    end
    # Warm actual full cases and both recursion/unique branches before five timings.
    for c in cases
        a,b,t,ok=sa_case(c,1)
        if !ok
            path=joinpath(ENV["STUDY_OUTPUT_DIR"],"failed_warmup.bin");write(path,vcat(a,b))
            println("STUDY_TRIAL {\"trial\":0,\"time_seconds\":null,\"correct\":false,\"checked_values\":$(c[1]),\"timing_valid\":false,\"warmup_failure\":true,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"int32\",\"count\":$(length(a)+length(b))}}")
            flush(stdout);exit(1)
        end
    end
    println("STUDY_SOFTWARE {\"CUDA_runtime\":$(ccall((:sa_runtime_version,SA_LIB),Cint,())),\"Thrust_version\":$(ccall((:sa_thrust_version,SA_LIB),Cint,())),\"CUB_version\":$(ccall((:sa_cub_version,SA_LIB),Cint,())),\"stream_kind\":\"legacy\",\"Julia_version\":\"$(VERSION)\",\"bridge_sha256\":\"$(bytes2hex(sha256(read(SA_LIB))))\",\"custom_GPU_definitions\":9}")
    for trial=1:5
        inputs=Int32[];outputs=Int32[];times=Float64[];ok=true;checked=0
        for c in cases
            a,b,t,good=sa_case(c,c[2]);append!(inputs,a);append!(outputs,b);push!(times,t);ok&=good;checked+=c[1]
        end
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$trial.bin");write(path,vcat(inputs,outputs))
        println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$(sum(times)),\"correct\":$ok,\"checked_values\":$checked,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_source_full_H2D_DC3_Thrust_allocation_sort_scan_rankfeedback_free_D2H_perconstruction_average\",\"timing_valid\":true,\"jit_excluded\":true,\"input_prefix_count\":$(length(inputs)),\"independent_certificate\":\"complete_permutation_and_finite_tail_lex_order\",\"source_internal_repeats_primary\":$(cases[1][2]),\"checked_final_construction_each_trial\":true,\"all_internal_iteration_outputs_archived\":false,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"int32\",\"count\":$(length(inputs)+length(outputs))},\"case_seconds\":[$(join(times,','))]}")
        flush(stdout);ok||exit(1)
    end
end
