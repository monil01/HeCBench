# Complete fixed source fixture/full horizon. No configurable ciphertext/search scope.
const NOUT=397*1024
function guarded(::Type{T},n) where T
    parent=CuArray(fill(reinterpret(T,UInt32(0xa7a7a7a7)),n+128))
    data=unsafe_wrap(CuArray,pointer(parent,65),(n,);own=false)
    (parent,data)
end
function guardbytes(buf)
    actual=Array(buf[1]);raw=reinterpret(UInt8,actual)
    guards=vcat(raw[1:256],raw[end-255:end]);all(==(0xa7),guards)||error("FullGPUallocationhead/tailguardfailure")
    guards
end
function independent(scores,cipher,out,diag)
    best=1;bestscore=999999f0;maxerr=0.
    for tid in 0:1023
        d=@view diag[tid*63+1:(tid+1)*63]
        for off in (0,30);sort(Int.(d[off+1:off+30]))==collect(0:29)||error("Permutationinvalid")end
        key=Int.(d[31:60]);order=sortperm(key);offset=zeros(Int,30);at=0
        for column in order;offset[column]=at;at+=13+(column<=7);end
        row=@view out[tid*397+1:(tid+1)*397];hist=zeros(Int,26);expectedhist=zeros(Int,26);fs=0f0;ds=0.
        for j in 0:396
            c=rem(j,30)+1;v=row[j+1];expected=cipher[offset[c]+div(j,30)+1]
            v==expected||error("Independentkeyplaintextmismatchtid=$tid pos=$j")
            0<=v<26||error("Invalidletter");hist[v+1]+=1;expectedhist[cipher[j+1]+1]+=1
            if j>0;term=scores[26*row[j]+v+1];fs+=term;ds+=Float64(term);end
        end
        hist==expectedhist||error("Letterhistogramchanged")
        actual=reinterpret(Float32,d[61]);actual==fs||error("F32scoreinconsistent")
        err=abs(Float64(actual)-ds);maxerr=max(maxerr,err);gamma=(396*2.0^-24)/(1-396*2.0^-24);err<=gamma*ds||error("Float64positive-sumerrorbound")
        if actual<bestscore;bestscore=actual;best=tid+1;end
    end
    plain=match(r"#define DECRYPTED_T \"([^\"]+)\"",read(joinpath(@__DIR__,"..","columnarSolver-cuda","main.cu"),String)).captures[1]
    actual=String(UInt8.(out[(best-1)*397+1:best*397].+Int32(97)));actual==plain||error("Fixedknownplaintextgate")
    best-1,maxerr
end
function main()
    source=read(joinpath(@__DIR__,"..","columnarSolver-cuda","main.cu"),String);enc=match(r"#define ENCRYPTED_T \"([^\"]+)\"",source).captures[1];cipher=Int32.(codeunits(enc)).-Int32(97)
    scores=zeros(Float32,676)
    for line in eachline(joinpath(@__DIR__,"..","columnarSolver-cuda","data"));k,v=split(line);scores[(Int(codeunit(k,1))-97)*26+Int(codeunit(k,2))-97+1]=parse(Float32,v);end
    buffers=(guarded(Float32,676),guarded(Int32,397),guarded(Int32,NOUT),guarded(UInt32,1024),guarded(UInt32,63*1024));gs,ge,go,gr,gd=map(last,buffers);copyto!(gs,scores);copyto!(ge,cipher)
    actualscores=Array(gs);actualcipher=Array(ge);actualscores==scores&&actualcipher==cipher||error("Actualinputbitschanged")
    gold=read(ARGS[1]);rawroot=ARGS[2];mkpath(rawroot)
    for trial in 0:5
        fill!(go,reinterpret(Int32,UInt32(0xa7a7a7a7)));fill!(gr,UInt32(0xa7a7a7a7));fill!(gd,UInt32(0xa7a7a7a7));CUDA.device_synchronize()
        start=time_ns();launch_both!(gs,ge,gr,go,gd);CUDA.device_synchronize();sec=(time_ns()-start)*1e-9
        seeds=Array(gr);out=Array(go);diag=Array(gd);guards=vcat(map(guardbytes,buffers)...)
        Array(gs)==actualscores&&Array(ge)==actualcipher||error("Readonlyinputchanged")
        raw=vcat(reinterpret(UInt8,seeds),reinterpret(UInt8,out),reinterpret(UInt8,diag));path=joinpath(rawroot,"trial$(trial)_state.bin");write(path,raw);write(joinpath(rawroot,"trial$(trial)_guards.bin"),guards)
        raw==gold||error("FullCPUrawsourcepathmismatchtrial=$trial saved=$path")
        best,err=independent(actualscores,actualcipher,out,diag)
        artifact=vcat(Float64.(actualscores),Float64.(actualcipher),Float64.(seeds),Float64.(out),Float64.(diag),Float64.(guards));name="output_trial$(trial).bin";write(name,artifact)
        @printf("%s{\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"source_setup_plus_full150000_decode_completed\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"raw_state_path\":\"%s\",\"guard_path\":\"%s\",\"source_CPU_exact\":true,\"known_plaintext_winner\":%d,\"reported_abs_error_definition\":\"Float32bigram_sum_vs_independentFloat64_sum\"}\n",trial>0 ? "STUDY_TRIAL " : "WARMUP ",trial,sec,length(artifact),err,name,length(artifact),path,joinpath(rawroot,"trial$(trial)_guards.bin"),best);flush(stdout)
    end
end
