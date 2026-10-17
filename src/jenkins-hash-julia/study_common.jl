using CUDA
using Printf
@inline rot64(x,k)=((x<<k)|(x>>(32-k)))&UInt64(0xffffffff)
function byte_oracle(words,len,seed)
    bytes=reinterpret(UInt8,words);remaining=Int(len);pos=1;mask=UInt64(0xffffffff);a=(UInt64(0xdeadbeef)+UInt64(len)+UInt64(seed))&mask;b=a;c=a
    word(q,n)=sum(UInt64(bytes[q+j])<<(8j) for j in 0:n-1;init=UInt64(0))
    while remaining>12
        a=(a+word(pos,4))&mask;b=(b+word(pos+4,4))&mask;c=(c+word(pos+8,4))&mask
        a=((a-c)&mask)⊻rot64(c,4);c=(c+b)&mask;b=((b-a)&mask)⊻rot64(a,6);a=(a+c)&mask;c=((c-b)&mask)⊻rot64(b,8);b=(b+a)&mask;a=((a-c)&mask)⊻rot64(c,16);c=(c+b)&mask;b=((b-a)&mask)⊻rot64(a,19);a=(a+c)&mask;c=((c-b)&mask)⊻rot64(b,4);b=(b+a)&mask
        remaining-=12;pos+=12
    end
    remaining==0&&return UInt32(c)
    a=(a+word(pos,min(remaining,4)))&mask;remaining>4&&(b=(b+word(pos+4,min(remaining-4,4)))&mask);remaining>8&&(c=(c+word(pos+8,remaining-8))&mask)
    c=((c⊻b)-rot64(b,14))&mask;a=((a⊻c)-rot64(c,11))&mask;b=((b⊻a)-rot64(a,25))&mask;c=((c⊻b)-rot64(b,16))&mask;a=((a⊻c)-rot64(c,4))&mask;b=((b⊻a)-rot64(a,14))&mask;c=((c⊻b)-rot64(b,24))&mask
    UInt32(c)
end
function study_main(prepare,args=ARGS)
    n=4099;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");sample=zeros(UInt8,64);str=codeunits("Four score and seven years ago");copyto!(sample,1,str,1,length(str));samplewords=collect(reinterpret(UInt32,sample));byte_oracle(samplewords,UInt32(30),UInt32(1))==0xcd628161||error("knownlookup3vector");fixtures=[];prefix=UInt32[]
    for kind in 0:1
        keys=kind==0 ? repeat_words(samplewords,n) : UInt32[UInt32((UInt64(i+1)*UInt64(2654435761)+UInt64(w)*UInt64(2246822519))&UInt64(0xffffffff)) for i in 0:n-1 for w in 0:15]
        lens=UInt32[UInt32(kind==0 ? i%61 : (i*17)%61) for i in 0:n-1];seeds=UInt32[kind==0 ? UInt32(i%2) : typemax(UInt32)-UInt32(i) for i in 0:n-1];expected=UInt32[byte_oracle(copy(keys[i*16+1:i*16+16]),lens[i+1],seeds[i+1]) for i in 0:n-1]
        for i in 0:n-1;lens[i+1]==0&&(expected[i+1]==0xdeadbeef+seeds[i+1]||error("zero-lengthproperty"));end
        append!(prefix,keys);append!(prefix,lens);append!(prefix,seeds);dk,dl,ds=CuArray(keys),CuArray(lens),CuArray(seeds);out=CUDA.zeros(UInt32,n);launch! = prepare(dl,ds,dk,out,n);launch!();push!(fixtures,(out,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=copy(prefix);seconds=0.0;ok=true
        for (out,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/repeat;got=Array(out);ok &=got==expected;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"jenkins_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"computed_hash_values\":8198,\"input_prefix_values\":147564,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_two_complete_lookup3_hash_fixtures\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
function repeat_words(words,n)
    v=Vector{UInt32}(undef,16n);for i in 0:n-1;copyto!(v,16i+1,words,1,16);end;v
end
