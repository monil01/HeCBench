using Printf
include("crypto_device.jl")
function cpu_check()
 root=@__DIR__;f=reinterpret(UInt32,read(joinpath(root,"fixture.bin")));gold=reinterpret(UInt32,read(joinpath(root,"schedule_oracle.bin")))
 salt=UInt8.(f[2:17]);padding=UInt8.(f[18:57]);words=zeros(UInt32,1048576*64)
 # One CPUlogical thread intentionally executes complete original gridstride work.
 kernel_w_block(salt,padding,words,UInt32(0),UInt32(1),UInt32(0),UInt32(1))
 @assert words==gold "GeneratedCPUfull64Mwordschedule mismatch"
 v=Int32.(words[1:4]);s=copy(words[1:4]);off=58
 for ci in 0:2
  len=f[off];off+=1;pw=copy(f[off:off+31]);off+=32;args=[]
  for n in [44,16,16,16,16];push!(args,UInt8.(f[off:off+n-1]));off+=n;end
  expected=f[off];off+=1;found=Int32[-1]
  decrypt_vmk_with_mac(UInt32(1),found,args...,v...,s...,pw,words,UInt32(0),UInt32(256),UInt32(0),UInt32(1))
  @assert reinterpret(UInt32,found[1])==expected "GeneratedCPUcomplete1MKDF/AESCCMcase$ci mismatch"
  println("GENERATED_CPU_COMPLETE case=",ci," length=",len," found=",found[1]);flush(stdout)
 end
 @assert off==length(f)+1
 println("GENERATED_CPU_PASS full_schedule_words=",length(words)," complete_1MKDF_AESCCM_cases=3")
end
cpu_check()
