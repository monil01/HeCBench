using Printf
const PARENT=@__DIR__
function loadbytes(path,expected)
    data=read(path);length(data)==expected||error("missing/shortfixture $path");data
end
function guarded(x::Vector{UInt8})
    y=fill(UInt8(0xa5),length(x)+128);copyto!(y,65,x,1,length(x));y
end
function allocbuf(x::Vector{UInt8},::Type{T}) where T
    bytes=guarded(x);length(bytes)%sizeof(T)==0||error("alignment");d=backend_array(collect(reinterpret(T,bytes)));g=div(64,sizeof(T));d,view(d,g+1:length(d)-g)
end
capture(d)=collect(reinterpret(UInt8,Array(d)))
resetoffsets!(O)=CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(O)),UInt8(255),640)
function main()
    length(ARGS)==1||error("caseset");set=ARGS[1];set in ("primary","partial_corrected")||error("set")
    cases=[parse.(Int,split(s)) for s in readlines(joinpath(PARENT,"cases_$(set).txt"))];length(cases)==(set=="primary" ? 33 : 18)||error("casesextent")
    kernels=nothing
    for trial=0:5
        actualpre=UInt8[];expectedpre=UInt8[];actualout=UInt8[];expectedout=UInt8[];comp=0.;dec=0.
        for (index,n0,dim0,cap,used,consumed) in cases
            dir=joinpath(PARENT,"fixtures",set,"case$index");input=loadbytes(joinpath(dir,"input_u64.bin"),n0*8);ec=loadbytes(joinpath(dir,"encoded_u64.bin"),cap*8);di=loadbytes(joinpath(dir,"decoder_input_u64.bin"),cap*8);ed=loadbytes(joinpath(dir,"decoded_u64.bin"),n0*8);oc=loadbytes(joinpath(dir,"offset_comp_i32.bin"),640);od=loadbytes(joinpath(dir,"offset_dec_i32.bin"),640)
            Ib,I=allocbuf(input,UInt64);Eb,E=allocbuf(zeros(UInt8,cap*8),UInt64);Db,D=allocbuf(zeros(UInt8,n0*8),UInt64);Ob,O=allocbuf(fill(UInt8(255),640),Int32)
            n=Int32(n0);dim=UInt8(dim0)
            if kernels===nothing;kernels=build_kernels(n,I,E,D,O,dim);end
            ipre=capture(Ib);append!(actualpre,ipre);append!(expectedpre,guarded(input));CUDA.device_synchronize()
            t=time_ns();resetoffsets!(O);kernels[1](n,I,E,O,dim);CUDA.device_synchronize();comp+=(time_ns()-t)*1e-9
            encoded=capture(Eb);co=capture(Ob);append!(actualout,encoded);append!(expectedout,guarded(ec))
            if encoded!=guarded(ec)||co!=guarded(oc)||capture(Ib)!=ipre
                ix=findfirst(encoded.!=guarded(ec));error("compression fullbytecheck case$index idx$ix")
            end
            header=(UInt64(n0)<<32)+UInt64(0x43504cff)+UInt64(dim0);copyto!(view(E,1:1),UInt64[header]);din=capture(Eb);append!(actualpre,din);append!(expectedpre,guarded(di));CUDA.device_synchronize()
            t=time_ns();resetoffsets!(O);kernels[2](E,D,O);CUDA.device_synchronize();dec+=(time_ns()-t)*1e-9
            append!(actualout,capture(Db));append!(expectedout,guarded(ed));append!(actualout,co);append!(expectedout,guarded(oc));append!(actualout,capture(Ob));append!(expectedout,guarded(od));capture(Eb)==din||error("decoderinputmutation")
        end
        if actualpre!=expectedpre || actualout!=expectedout
            ix=findfirst(actualout.!=expectedout);error("fullbytecheck final idx$ix")
        end
        if trial>0
            path=joinpath(ENV["STUDY_OUTPUT_DIR"],"mpc_trial$(trial).bin");open(path,"w") do f;write(f,actualpre);write(f,actualout);end
            @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"compression_seconds\":%.17g,\"decompression_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum_source_offsetreset_compress_completion_and_offsetreset_decompress_completion_excludes_header_hostfeedback_readbacks_capture\",\"iterations\":%d,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,comp+dec,comp,dec,length(actualpre)+length(actualout),length(cases),path,length(actualpre)+length(actualout));flush(stdout)
        end
    end
end
