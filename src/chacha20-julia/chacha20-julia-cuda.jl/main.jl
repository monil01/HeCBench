using CUDA
using Printf
using StaticArrays

# Julia port of chacha20-cuda benchmark.
# One kernel: generates keystream (equivalent to XOR with zero plaintext) and
# writes it to the result buffer. Parallelized per 64-byte block.

@inline function rotl32(x::UInt32, n::Int32)
    return (x << n) | (x >> (Int32(32) - n))
end

@inline function pack4_le(a0::UInt8, a1::UInt8, a2::UInt8, a3::UInt8)
    return UInt32(a0) | (UInt32(a1) << 8) | (UInt32(a2) << 16) | (UInt32(a3) << 24)
end

@inline function chacha_block!(key_words,nonce_words,out_bytes,counter::UInt64)
    # state[1..16]
    s = @MArray zeros(UInt32, 16)
    # magic constant "expand 32-byte k" packed little-endian
    s[1]  = UInt32(0x61707865)
    s[2]  = UInt32(0x3320646e)
    s[3]  = UInt32(0x79622d32)
    s[4]  = UInt32(0x6b206574)
    @inbounds for i in 1:8
        s[i + 4] = key_words[i]
    end
    s[13] = UInt32(counter & UInt64(0xFFFFFFFF))
    s[14] = UInt32((counter >> 32) & UInt64(0xFFFFFFFF))
    s[15] = nonce_words[1]
    s[16] = nonce_words[2]

    r = @MArray zeros(UInt32, 16)
    @inbounds for i in 1:16
        r[i] = s[i]
    end

    @inbounds for _ in 1:10
        # QUARTERROUND(a,b,c,d)  (0-based CUDA indices, we +1 for Julia)
        # (0,4,8,12)
        r[1] += r[5]; r[13] = rotl32(r[13] ⊻ r[1], Int32(16))
        r[9] += r[13]; r[5] = rotl32(r[5] ⊻ r[9], Int32(12))
        r[1] += r[5]; r[13] = rotl32(r[13] ⊻ r[1], Int32(8))
        r[9] += r[13]; r[5] = rotl32(r[5] ⊻ r[9], Int32(7))
        # (1,5,9,13)
        r[2] += r[6]; r[14] = rotl32(r[14] ⊻ r[2], Int32(16))
        r[10] += r[14]; r[6] = rotl32(r[6] ⊻ r[10], Int32(12))
        r[2] += r[6]; r[14] = rotl32(r[14] ⊻ r[2], Int32(8))
        r[10] += r[14]; r[6] = rotl32(r[6] ⊻ r[10], Int32(7))
        # (2,6,10,14)
        r[3] += r[7]; r[15] = rotl32(r[15] ⊻ r[3], Int32(16))
        r[11] += r[15]; r[7] = rotl32(r[7] ⊻ r[11], Int32(12))
        r[3] += r[7]; r[15] = rotl32(r[15] ⊻ r[3], Int32(8))
        r[11] += r[15]; r[7] = rotl32(r[7] ⊻ r[11], Int32(7))
        # (3,7,11,15)
        r[4] += r[8]; r[16] = rotl32(r[16] ⊻ r[4], Int32(16))
        r[12] += r[16]; r[8] = rotl32(r[8] ⊻ r[12], Int32(12))
        r[4] += r[8]; r[16] = rotl32(r[16] ⊻ r[4], Int32(8))
        r[12] += r[16]; r[8] = rotl32(r[8] ⊻ r[12], Int32(7))
        # (0,5,10,15)
        r[1] += r[6]; r[16] = rotl32(r[16] ⊻ r[1], Int32(16))
        r[11] += r[16]; r[6] = rotl32(r[6] ⊻ r[11], Int32(12))
        r[1] += r[6]; r[16] = rotl32(r[16] ⊻ r[1], Int32(8))
        r[11] += r[16]; r[6] = rotl32(r[6] ⊻ r[11], Int32(7))
        # (1,6,11,12)
        r[2] += r[7]; r[13] = rotl32(r[13] ⊻ r[2], Int32(16))
        r[12] += r[13]; r[7] = rotl32(r[7] ⊻ r[12], Int32(12))
        r[2] += r[7]; r[13] = rotl32(r[13] ⊻ r[2], Int32(8))
        r[12] += r[13]; r[7] = rotl32(r[7] ⊻ r[12], Int32(7))
        # (2,7,8,13)
        r[3] += r[8]; r[14] = rotl32(r[14] ⊻ r[3], Int32(16))
        r[9] += r[14]; r[8] = rotl32(r[8] ⊻ r[9], Int32(12))
        r[3] += r[8]; r[14] = rotl32(r[14] ⊻ r[3], Int32(8))
        r[9] += r[14]; r[8] = rotl32(r[8] ⊻ r[9], Int32(7))
        # (3,4,9,14)
        r[4] += r[5]; r[15] = rotl32(r[15] ⊻ r[4], Int32(16))
        r[10] += r[15]; r[5] = rotl32(r[5] ⊻ r[10], Int32(12))
        r[4] += r[5]; r[15] = rotl32(r[15] ⊻ r[4], Int32(8))
        r[10] += r[15]; r[5] = rotl32(r[5] ⊻ r[10], Int32(7))
    end

    @inbounds for i in 1:16
        r[i] += s[i]
    end

    # unpack r[1..16] into out_bytes[block_start+1 .. +64] little-endian
    base = Int32(counter) * Int32(64)
    @inbounds for i in 1:16
        w = r[i]
        b0 = UInt8(w & UInt32(0xff))
        b1 = UInt8((w >> 8) & UInt32(0xff))
        b2 = UInt8((w >> 16) & UInt32(0xff))
        b3 = UInt8((w >> 24) & UInt32(0xff))
        # only write within bounds; caller passes num_bytes >= num_blocks*64 already
        idx = base + Int32((i - 1) * 4)
        out_bytes[idx + 1] = b0
        out_bytes[idx + 2] = b1
        out_bytes[idx + 3] = b2
        out_bytes[idx + 4] = b3
    end
    return
end

const H_KEY = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
const H_NONCE = "0001020304050607"
const H_KEYSTREAM = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9"

function hex_to_bytes(s::AbstractString)
    n = length(s) ÷ 2
    b = Vector{UInt8}(undef, n)
    for i in 1:n
        b[i] = parse(UInt8, s[2i-1:2i], base=16)
    end
    return b
end

function study_kernel!(textkey,textnonce,textref,lookup,rawkey,rawnonce,rawref,result,nbytes)
 t=Int(threadIdx().x)
 if t<=32;rawkey[t]=(lookup[Int(textkey[2t-1])+1]<<4)|lookup[Int(textkey[2t])+1];end
 if t<=8;rawnonce[t]=(lookup[Int(textnonce[2t-1])+1]<<4)|lookup[Int(textnonce[2t])+1];end
 i=t
 while i<=nbytes
  rawref[i]=(lookup[Int(textref[2i-1])+1]<<4)|lookup[Int(textref[2i])+1];i+=256
 end
 CUDA.sync_threads()
 if t==1
  keywords=@MArray zeros(UInt32,8);noncewords=@MArray zeros(UInt32,2)
  for j in 1:8;keywords[j]=pack4_le(rawkey[4j-3],rawkey[4j-2],rawkey[4j-1],rawkey[4j]);end
  for j in 1:2;noncewords[j]=pack4_le(rawnonce[4j-3],rawnonce[4j-2],rawnonce[4j-1],rawnonce[4j]);end
  for counter in 0:div(nbytes,64)-1;chacha_block!(keywords,noncewords,result,UInt64(counter));end
 end
end
function launch(args...)
 @cuda threads=256 blocks=1 study_kernel!(args...)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 repeat=length(args)>0 ? parse(Int,args[1]) : 10;reference=hex_to_bytes(H_KEYSTREAM);nbytes=length(reference)
 lookup=zeros(UInt8,256);for i in 0:9;lookup[Int('0')+i+1]=UInt8(i);end;for i in 0:25;lookup[Int('a')+i+1]=UInt8(i+10);lookup[Int('A')+i+1]=UInt8(i+10);end
 tk=CuArray(collect(codeunits(H_KEY)));tn=CuArray(collect(codeunits(H_NONCE)));tr=CuArray(collect(codeunits(H_KEYSTREAM)));table=CuArray(lookup);rk=CUDA.zeros(UInt8,32);rn=CUDA.zeros(UInt8,8);rr=CUDA.zeros(UInt8,nbytes);result=CUDA.zeros(UInt8,nbytes)
 expected=Float64.(vcat(reference,hex_to_bytes(H_KEY),hex_to_bytes(H_NONCE),reference))
 launch(tk,tn,tr,table,rk,rn,rr,result,nbytes);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns()
  for _ in 1:repeat
   fill!(result,UInt8(0));launch(tk,tn,tr,table,rk,rn,rr,result,nbytes)
  end
  CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat;values=Float64.(vcat(Array(result),Array(rk),Array(rn),Array(rr)))
  emit(trial,seconds,values,expected,repeat;case_times=[seconds])
 end
end
main(ARGS)
