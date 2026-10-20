using Printf
const STUDY_DIR=joinpath(@__DIR__)
mutable struct StudyBuffer{T,A,V}
    base::A
    data::V
    n::Int
    guard::Int
end
function study_buffer(::Type{T},n) where T
    g=div(16,sizeof(T));base=alloc_backend(T,n+2g)
    pattern=reinterpret(T,fill(UInt8(0xa5),sizeof(T)))[1]
    fill!(base,pattern);data=view(base,g+1:g+n);fill!(data,zero(T))
    return StudyBuffer{T,typeof(base),typeof(data)}(base,data,n,g)
end
function raw_bytes(b::StudyBuffer)
    return collect(reinterpret(UInt8,Array(b.base)))
end
function live_bytes(b::StudyBuffer,n)
    0<=n<=b.n||error("livebounds")
    a=Array(b.base);return collect(reinterpret(UInt8,vcat(a[1:b.guard],a[b.guard+1:b.guard+n],a[end-b.guard+1:end])))
end
function free_buffer!(b);CUDA.unsafe_free!(b.base);end
function exclusive_scan!(input,out,n,temps)
    groups=cld(n,512);sums=alloc_backend(UInt32,groups);push!(temps,sums)
    run_scan!(input,out,sums,n,groups)
    if groups>1
        offsets=alloc_backend(UInt32,groups);push!(temps,offsets)
        exclusive_scan!(sums,offsets,groups,temps);run_add!(out,offsets,n)
    end
end
function main()
    input=reinterpret(UInt32,read(joinpath(STUDY_DIR,"study_input_u32.bin")))
    expectedpre=read(joinpath(STUDY_DIR,"oracle_prefix_u8.bin"));expectedout=read(joinpath(STUDY_DIR,"oracle_output_u8.bin"))
    cases=[parse.(Int,split(line)) for line in readlines(joinpath(STUDY_DIR,"cases.txt"))]
    for trial in 0:5
        actualpre=UInt8[];actualout=UInt8[];ip=0;comp=0.0;decomp=0.0
        sizehint!(actualpre,length(expectedpre));sizehint!(actualout,length(expectedout))
        for (original,nb,nf,nd) in cases
            words=nb*512;inp=study_buffer(UInt32,words);decoded=study_buffer(UInt32,words)
            fs=study_buffer(UInt32,nb+1);ds=study_buffer(UInt32,nb+1);fo=study_buffer(UInt32,nb+1);doo=study_buffer(UInt32,nb+1)
            tf=study_buffer(UInt8,nb*64);td=study_buffer(UInt8,nb*2048);pf=study_buffer(UInt8,nb*64);pd=study_buffer(UInt8,nb*2048)
            copyto!(inp.data,collect(view(input,ip+1:ip+words)));ip+=words
            append!(actualpre,raw_bytes(inp));CUDA.device_synchronize()
            t0=time_ns();run_match!(inp.data,fs.data,ds.data,tf.data,td.data,nb)
            temps=Any[];exclusive_scan!(fs.data,fo.data,nb+1,temps);exclusive_scan!(ds.data,doo.data,nb+1,temps)
            run_pack!(fo.data,doo.data,tf.data,td.data,pf.data,pd.data,nb);CUDA.device_synchronize()
            for a in temps;CUDA.unsafe_free!(a);end
            t1=time_ns();run_decode!(decoded.data,fo.data,doo.data,pf.data,pd.data,nb);CUDA.device_synchronize();t2=time_ns()
            comp+=(t1-t0)*1e-9;decomp+=(t2-t1)*1e-9
            for b in (decoded,fs,ds,fo,doo,tf,td);append!(actualout,raw_bytes(b));end
            append!(actualout,live_bytes(pf,nf));append!(actualout,live_bytes(pd,nd))
            for b in (inp,decoded,fs,ds,fo,doo,tf,td,pf,pd);free_buffer!(b);end
        end
        ip==length(input)||error("input extent")
        actualpre==expectedpre||error("actual device input/guard mismatch")
        if actualout!=expectedout
            j=findfirst(i->actualout[i]!=expectedout[i],eachindex(expectedout))
            error("full codec byte mismatch at $j actual=$(actualout[j]) expected=$(expectedout[j])")
        end
        if trial>0
            path=joinpath(ENV["STUDY_OUTPUT_DIR"],"lzss_trial$(trial).bin")
            open(path,"w")do io;write(io,actualpre);write(io,actualout);end
            count=length(actualpre)+length(actualout)
            @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"compression_seconds\":%.17g,\"decompression_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum6cases_source_compression_localmatch_globalexclusive_scans_pack_scratch_and_decompression_completion_excludes_captures\",\"iterations\":6,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,comp+decomp,comp,decomp,count,path,count)
            flush(stdout)
        end
    end
end
