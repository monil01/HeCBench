using Printf
const PERLIN_HASH=Int32[151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180]
const PERLIN_CASES=((512,1,250f0,Int32(0),Int32(3)),(512,2,250f0,Int32(0),Int32(3)),(512,4,250f0,Int32(0),Int32(3)),(512,8,250f0,Int32(0),Int32(3)),(512,16,250f0,Int32(0),Int32(3)),(512,32,250f0,Int32(0),Int32(3)),(16,1,1f0,Int32(7),Int32(3)),(48,2,250f0,Int32(5),Int32(1)))
function perlin_guarded(n,init,alloc)
    h=fill(UInt8(173),n+512);fill!(view(h,257:n+256),UInt8(init))
    d=alloc(h)
    UInt(pointer(d))%256==0 || error("base alignment")
    (device=d,payload=view(d,257:n+256))
end
function perlin_constants(hash,gx,gy)
    v=UInt8[]
    append!(v,reinterpret(UInt8,Array(hash)));append!(v,reinterpret(UInt8,Array(gx)));append!(v,reinterpret(UInt8,Array(gy)));v
end
function perlin_execute(trial,alloc,launch,observe)
    prefix=UInt8[];post=UInt8[];seconds=0.0
    if trial>0;sizehint!(prefix,770720544);sizehint!(post,770720544);end
    q=Float32(.70710678118654752440)
    hash=alloc(PERLIN_HASH);gx=alloc(Float32[1,-1,0,0,q,-q,q,-q]);gy=alloc(Float32[0,0,1,-1,q,q,-q,-q])
    for (height,ns,ppu,seed,octaves) in PERLIN_CASES
        p=NoiseParameters(ppu,seed,octaves,2f0,.5f0)
        out=perlin_guarded(4*61440*height,17,alloc);param=perlin_guarded(36,0,alloc)
        streams=[CUDA.CuStream() for _ in 1:ns]
        observe(param.payload,p,Int32(height),Int32(ns));CUDA.device_synchronize()
        if trial>0
            append!(prefix,perlin_constants(hash,gx,gy));append!(prefix,Array(param.device));append!(prefix,Array(out.device))
        end
        ph=Int32(height÷ns);CUDA.device_synchronize();t=time_ns()
        for i in 0:ns-1
            launch(streams[i+1],Int32(i)*ph,ph,p,hash,gx,gy,out.payload)
        end
        CUDA.device_synchronize();seconds+=(time_ns()-t)*1e-9
        if trial>0
            append!(post,perlin_constants(hash,gx,gy));append!(post,Array(param.device));append!(post,Array(out.device))
        end
        CUDA.unsafe_free!(out.device);CUDA.unsafe_free!(param.device)
    end
    for a in (hash,gx,gy);CUDA.unsafe_free!(a);end
    if trial>0
        length(prefix)==770720544 && length(post)==770720544 || error("artifact count")
        name="perlin_trial$(trial).bin"
        open(name,"w") do io;write(io,prefix);write(io,post);end
        ae=Ref(0.0);re=Ref(0.0)
        good=ccall((:perlin_check,joinpath(@__DIR__,"libstudy_check.so")),Cint,(Cstring,Ref{Cdouble},Ref{Cdouble}),name,ae,re)
        good==1 || error("independent full physical field audit")
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":1541441088,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"timing_region\":\"sum8_named_perlin_source16x16_multistream_launch_wholecontextcompletion\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":1541441088,\"byte_order\":\"little\"}}\n",trial,seconds,ae[],re[],name)
        flush(stdout)
    end
end
function perlin_study(alloc,launch,observe)
    CUDA.allowscalar(false)
    println("\nPERLIN_STUDY_BEGIN originalwidth61440 namedstrips512_16_48");flush(stdout)
    perlin_execute(0,alloc,launch,observe)
    for trial in 1:5;perlin_execute(trial,alloc,launch,observe);GC.gc();end
end
