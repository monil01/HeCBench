function bitcracker_study(schedule!,decrypt!)
 root=@__DIR__;f=reinterpret(UInt32,read(joinpath(root,"fixture.bin")));@assert length(f)==483
 guarded(v)=vcat(fill(eltype(v)(eltype(v)==UInt8 ? 0xa5 : eltype(v)==Int32 ? reinterpret(Int32,UInt32(0xa5a5a5a5)) : 0xa5a5a5a5),32),v,fill(eltype(v)(eltype(v)==UInt8 ? 0xa5 : eltype(v)==Int32 ? reinterpret(Int32,UInt32(0xa5a5a5a5)) : 0xa5a5a5a5),32))
 append_wire!(wire,v)=append!(wire,eltype(v)==Int32 ? reinterpret(UInt32,v) : v)
 for trial in 0:5
  prefix=UInt32[3,1048576];sizehint!(prefix,67112675);extra=UInt32[];seconds=0.;times=Float64[];ok=true
  salt_h=guarded(UInt8.(f[2:17]));padding_h=guarded(UInt8.(f[18:57]));w_h=guarded(zeros(UInt32,1048576*64))
  salt_b=CuArray(salt_h);padding_b=CuArray(padding_h);w_b=CuArray(w_h);salt=view(salt_b,33:48);padding=view(padding_b,33:72);w=view(w_b,33:length(w_b)-32)
  sa=Array(salt_b);pa=Array(padding_b);ok&=sa==salt_h&&pa==padding_h;append_wire!(prefix,sa);append_wire!(prefix,pa)
  CUDA.synchronize(CUDA.context());start=time_ns();schedule!(salt,padding,w);CUDA.synchronize(CUDA.context());table_seconds=(time_ns()-start)*1e-9;seconds+=table_seconds
  words=Array(w_b);ok&=words[1:32]==w_h[1:32]&&words[end-31:end]==w_h[end-31:end];append_wire!(prefix,words)
  ok&=Array(salt_b)==salt_h&&Array(padding_b)==padding_h
  v=Int32.(words[33:36]);s=copy(words[33:36]);off=58
  for ci in 0:2
   lengthpw=f[off];off+=1;push!(prefix,lengthpw);append_wire!(prefix,reinterpret(UInt32,v));append_wire!(prefix,s)
   initial=Any[guarded(copy(f[off:off+31]))];off+=32
   for n in [44,16,16,16,16];push!(initial,guarded(UInt8.(f[off:off+n-1])));off+=n;end
   expected=f[off];off+=1;push!(initial,guarded(Int32[-1]));base=CuArray.(initial);buffers=map(a->view(a,33:length(a)-32),base);pw,vmk,iv,mac,maciv,computed,found=buffers
   for (b,h)in zip(base,initial);z=Array(b);ok&=z==h;append_wire!(prefix,z);end
   CUDA.synchronize(CUDA.context());start=time_ns();decrypt!(found,vmk,iv,mac,maciv,computed,v,s,pw,w);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;seconds+=dt;push!(times,dt)
   for j in eachindex(base)
    z=Array(base[j]);h=initial[j];ok&=z[1:32]==h[1:32]&&z[end-31:end]==h[end-31:end]
    if j!=7;ok&=z==h;else;ok&=reinterpret(UInt32,z[33])==expected;end
    append_wire!(extra,z)
   end
   for b in base;CUDA.unsafe_free!(b);end
  end
  @assert off==length(f)+1&&length(prefix)==67110908&&length(extra)==1767
  after=Array(w_b);ok&=after==words
  for b in [salt_b,padding_b,w_b];CUDA.unsafe_free!(b);end
  if trial==0;ok||error("Bitcrackerwarmupfulloracle/guardsfailure");continue;end
  append!(prefix,extra);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,prefix)
  ok&=success(pipeline(Cmd(["python3",joinpath(root,"audit_native_outputs.py"),path]),stdout=devnull,stderr=stderr))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"schedule_seconds\":%.17g,\"case_times_seconds\":[%.17g,%.17g,%.17g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"full1Mschedule_plus3full1MKDF_AESCCM_knownsynthetic_singlecandidate_kernel_launches_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"table_readonly_full_equal\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,table_seconds,times[1],times[2],times[3],ok ? "true" : "false",length(prefix),path,length(prefix));flush(stdout)
  ok||error("Fullindependent1Mschedule_1MKDF_AESCCM_failed")
 end
end
