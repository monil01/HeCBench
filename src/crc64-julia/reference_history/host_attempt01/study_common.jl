using CUDA,Printf
const ROOT=@__DIR__
const TABLE=collect(reinterpret(UInt64,read(joinpath(ROOT,"crc64_table.bin"))))
const INTER=collect(reinterpret(UInt64,read(joinpath(ROOT,"crc64_interleaved_table.bin"))))
const POWERS=collect(reinterpret(UInt64,read(joinpath(ROOT,"crc64_x_pow_2n.bin"))))
const INPUTS=[read(joinpath(ROOT,"study_input$(i)_u8.bin")) for i in 1:2]
const ORACLE=collect(reinterpret(UInt64,read(joinpath(ROOT,"oracle_u64.bin"))))
const SENTINEL=UInt64(0x5a5a5a5a5a5a5a5a)
devicewords(d)=CUDA.CuDeviceArray{UInt32,1,1}(reinterpret(CUDA.LLVMPtr{UInt32,1},pointer(CUDA.cudaconvert(d))),(div(length(d),4),))
function host_crc(data)
 words=reinterpret(UInt32,view(data,1:4div(length(data),4)))
 crc_interleaved(data,words,0,length(data),TABLE,INTER)
end
function combine_parts(cs,sz)
 v=cs[1]
 for i in 2:64;v=crc_combine(v,cs[i],sz[i],POWERS);end
 v
end
function parallel_crc(data,launch)
 n=length(data)
 n<=2048&&return host_crc(data)
 sz=CuArray{UInt64}(undef,64);cs=CuArray{UInt64}(undef,64);d=CuArray{UInt8}(undef,n);t=CuArray{UInt64}(undef,1024);it=CuArray{UInt64}(undef,1024)
 copyto!(d,data);copyto!(t,TABLE);copyto!(it,INTER);words=devicewords(d)
 GC.@preserve d begin
  launch(sz,cs,d,words,t,it,Int64(n))
  sizes=Array(sz);checks=Array(cs);v=combine_parts(checks,sizes)
 end
 CUDA.unsafe_free!(sz);CUDA.unsafe_free!(cs);CUDA.unsafe_free!(d);CUDA.unsafe_free!(t);CUDA.unsafe_free!(it)
 v
end
function crc_study(launch)
 allok=true
 # Warm the complete host/device API and both dtype-identical length cases before timing.
 for input in INPUTS;parallel_crc(input,launch);end
 host_crc(collect(codeunits("123456789")));CUDA.device_synchronize(blocking=true)
 for trial in 1:5
  prefix=UInt64[];results=UInt64[];guards=UInt64[];times=zeros(2)
  for (ci,data) in enumerate(INPUTS)
   n=length(data);actuals=Vector{UInt64}(undef,10);CUDA.device_synchronize(blocking=true);start=time_ns()
   for j in 1:10;actuals[j]=parallel_crc(data,launch);end
   CUDA.device_synchronize(blocking=true);times[ci]=(time_ns()-start)*1e-9/10
   # Untimed state/input capture through unchanged device algorithm with guarded output arrays.
   d=CuArray(data);t=CuArray(TABLE);it=CuArray(INTER);sb=CUDA.fill(SENTINEL,128);cb=CUDA.fill(SENTINEL,128);sz=view(sb,33:96);cs=view(cb,33:96);words=devicewords(d)
   GC.@preserve d launch(sz,cs,d,words,t,it,Int64(n));CUDA.device_synchronize(blocking=true)
   sizes=Array(sz);checks=Array(cs);append!(prefix,UInt64.(Array(d)))
   if ci==2;append!(prefix,Array(t));append!(prefix,Array(it));end
   rawsz=Array(sb);rawcs=Array(cb);append!(guards,rawsz[1:32],rawsz[97:128],rawcs[1:32],rawcs[97:128])
   checkbytes=collect(reinterpret(UInt8,[actuals[end] ⊻ typemax(UInt64)]));inverted=host_crc(vcat(data,checkbytes));split=12345;merged=crc_combine(host_crc(data[1:split]),host_crc(data[split+1:end]),n-split,POWERS)
   append!(results,sizes,checks,actuals);push!(results,inverted,merged)
   for a in (d,t,it,sb,cb);CUDA.unsafe_free!(a);end
  end
  known=collect(codeunits("123456789"));push!(results,parallel_crc(known,launch));append!(prefix,POWERS,UInt64.(known),UInt64[length(INPUTS[1]),length(INPUTS[2]),64,10,9])
  ok=results==ORACLE&&all(==(SENTINEL),guards);allok&=ok;artifact=vcat(prefix,results,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":537,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"sum_two_original_parallelCRC_API_allocation_copy_kernel_readback_CPUcombine_free_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_cases\":2,\"component_time_seconds\":{\"bytes65555\":%.12g,\"bytes65677\":%.12g},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ok ? "0" : "null",times[1],times[2],path,length(artifact))
 end
 allok||error("CRC/API/partial/checkbytes/combine/guard mismatch");println("PASS")
end
