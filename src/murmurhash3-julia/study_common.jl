using CUDA,Printf
function hash_study(factory)
 keys=read(joinpath(@__DIR__,"keys_u8.bin"));offsets=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"offsets_u32.bin"))));lengths=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"lengths_u32.bin"))));gold=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"oracle_u64.bin"))));G=16;BG=UInt8(0xa5);WG=UInt32(0xa55a3cc3);OG=UInt64(0xcafebabedeadbeef);R=10;allok=true
 # Fixture boundaries mirror the generator; input bytes themselves remain external/hash-bound.
 cases=Tuple{Int,Int,Int,Int,Int,Int}[];bo=0;oo=0;lo=0;go=0
 special=vcat(collect(0:47),[63,64,65,255,256,257,4095,4096,4097,9999])
 for n in [257,4099],fixture in [0,1]
  lens=[i<length(special) ? special[i+1] : (i*61+fixture*17)%257 for i in 0:n-1];bytes=sum(lens)+(fixture==0 ? 0 : sum(i*3%7 for i in 0:n-1));push!(cases,(n,bo,bytes,oo,lo,go));bo+=bytes;oo+=n+1;lo+=n;go+=2n
 end
 for trial in 0:5
  prefix=UInt64[];outputs=UInt64[];times=Float64[];ok=true;bad=0
  for (n,bo,bytes,oo,lo,go) in cases
   kh=vcat(fill(BG,G),keys[bo+1:bo+bytes],fill(BG,G));oh=vcat(fill(WG,G),offsets[oo+1:oo+n+1],fill(WG,G));lh=vcat(fill(WG,G),lengths[lo+1:lo+n],fill(WG,G));initial=fill(OG,2n+2G);dk=CuArray(kh);dof=CuArray(oh);dl=CuArray(lh);out=CuArray(initial);launch! =factory(view(dk,G+1:G+bytes),view(dof,G+1:G+n+1),view(dl,G+1:G+n),view(out,G+1:G+2n),n)
   if trial==0;launch!();CUDA.device_synchronize(blocking=true);continue;end
   before=Array(out);ok &=before==initial;CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:R;launch!();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/R)
   ak=Array(dk);ao=Array(dof);al=Array(dl);answer=Array(out);ok &=ak==kh&&ao==oh&&al==lh;bad+=count(!=(OG),answer[1:G])+count(!=(OG),answer[end-G+1:end])+count(answer[G+1:G+2n].!=gold[go+1:go+2n]);ok &=answer[G+1]==0&&answer[G+2]==0
   append!(prefix,UInt64.(ak));append!(prefix,UInt64.(ao));append!(prefix,UInt64.(al));append!(prefix,before);append!(outputs,answer)
  end
  trial==0&&continue
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%s,\"timing_region\":\"sum4_hash_kernel_hostwindows_R10\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),bad==0 ? 0 : 1,bad==0 ? "0" : "null",join(times,","),path,length(artifact))
 end
 allok||error("Full128bithash/input/guard/emptyseedcheck failed");println("PASS")
end
