using Printf
include("reference_history/cuda_jl_candidate01_cpu_mismatch/crypto_device.jl")
function diagnose()
 text=read(joinpath(@__DIR__,"reference_history","cuda_jl_candidate01_cpu_mismatch","crypto_device.jl"),String);start=findfirst("@inline function decrypt_vmk_with_mac",text).start;stop=findfirst("@inline function kernel_w_block",text).start-1;body=text[start:stop]
 body=replace(body,"decrypt_vmk_with_mac("=>"diagnose_kdf(")
 marker=findfirst(r"  a_c\d+ = \(\(bc_load_u32_le\(vmkIV",body).start
 # Actual alpha-renamedhash local identities bound to this preserved candidate.
 body=body[1:marker-1]*"  return (hash0_c41,hash1_c42,hash2_c43,hash3_c44,hash4_c45,hash5_c46,hash6_c47,hash7_c48)\n"*body[marker:end]
 Base.include_string(Main,body,"diagnostic_kdf_from_candidate01")
 # FIPS-197 AES256 known-answer; key bigendian and message/output littleendian
 key=UInt32[0x00010203,0x04050607,0x08090a0b,0x0c0d0e0f,0x10111213,0x14151617,0x18191a1b,0x1c1d1e1f]
 block=UInt32[0x33221100,0x77665544,0xbbaa9988,0xffeeddcc]
 result=encrypt(key...,block...);println("AES256_FIPS_WORDS=",string.(collect(result),base=16,pad=8));flush(stdout)
 f=reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin")));words=reinterpret(UInt32,read(joinpath(@__DIR__,"schedule_oracle.bin")));off=59;pw=copy(f[off:off+31]);off+=32;args=[]
 for n in [44,16,16,16,16];push!(args,UInt8.(f[off:off+n-1]));off+=n;end
 got=Base.invokelatest(diagnose_kdf,UInt32(1),Int32[-1],args...,Int32.(words[1:4])...,words[1:4]...,pw,words,UInt32(0),UInt32(256),UInt32(0),UInt32(1));println("KDF_WORDS=",string.(collect(got),base=16,pad=8))
end
diagnose()
