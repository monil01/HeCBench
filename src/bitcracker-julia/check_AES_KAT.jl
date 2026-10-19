include("crypto_device.jl")
key=UInt32[0x00010203,0x04050607,0x08090a0b,0x0c0d0e0f,0x10111213,0x14151617,0x18191a1b,0x1c1d1e1f]
message=UInt32[0x33221100,0x77665544,0xbbaa9988,0xffeeddcc]
got=collect(encrypt(key...,message...));hex=bytes2hex(reinterpret(UInt8,got));@assert hex=="8ea2b7ca516745bfeafc49904b496089"
println("{\"repaired_generated_AES256_FIPS_KAT\":true,\"ciphertext\":\"",hex,"\",\"scope\":\"CPUonly immutable same helper as accepted GPU sources; no extra GPU trial\"}")
