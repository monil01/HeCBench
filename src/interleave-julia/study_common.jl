using CUDA,Printf
@generated function iv_accumulate!(src,dst,i,::Val{LAYOUT}) where LAYOUT
 indices=[gensym(:index) for _ in 1:16]
 setup=[:($(indices[f])=Int32(16)+(LAYOUT==0 ? (i-Int32(1))*Int32(16)+Int32($f) : Int32($(f-1))*Int32(4096)+i)) for f in 1:16]
 # Same16 explicit field updates and sourceCOUNT4096 as native kernel bodies.
 updates=[:(@inbounds dst[$(indices[f])]=dst[$(indices[f])]+src[$(indices[f])]) for f in 1:16]
 quote
  $(setup...)
  for iteration in 1:4096
   $(updates...)
  end
  return
 end
end
function interleave_study(factory)
 P=4096;F=16;G=16;W=P*F+2G;guard=UInt32(0xA55A3CC3);sf=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture_source_aos_u32.bin"))));df=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture_initial_aos_u32.bin"))));src=CUDA.fill(guard,W);dst=CUDA.fill(guard,W);launch! =factory(src,dst);s=fill(guard,W);d=fill(guard,W)
 function prepare!(fixture,layout)
  fill!(s,guard);fill!(d,guard)
  for i in 1:P,f in 1:F
   index=G+(layout==0 ? (i-1)*F+f : (f-1)*P+i);origin=fixture*P*F+(i-1)*F+f;s[index]=sf[origin];d[index]=df[origin]
  end
  copyto!(src,s);copyto!(dst,d);CUDA.device_synchronize(blocking=true)
 end
 for layout in 0:1;prepare!(0,layout);launch!(Int32(P),Val(layout));CUDA.device_synchronize(blocking=true);end
 allok=true
 for trial in 1:5
  prefix=UInt32[];outputs=UInt32[];times=Float64[];ok=true;bad=0
  for fixture in 0:1,shape in 0:1,layout in 0:1
   n=shape==0 ? 4096 : 4093;prepare!(fixture,layout);initial=Array(dst);ok &=initial==d;start=time_ns();launch!(Int32(n),Val(layout));CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9);actualsrc=Array(src);out=Array(dst);ok &=actualsrc==s
   for i in 1:G;bad+=out[i]!=guard||out[W-G+i]!=guard;end
   for i in 1:P,f in 1:F
    index=G+(layout==0 ? (i-1)*F+f : (f-1)*P+i);origin=fixture*P*F+(i-1)*F+f;expected=df[origin]+(i<=n ? UInt32(4096)*sf[origin] : UInt32(0));bad+=out[index]!=expected
   end
   append!(prefix,actualsrc);append!(prefix,initial);append!(outputs,out)
  end
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum8_separate_layout_kernel_hostwindows\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),bad==0 ? 0 : 1,join(times,","),path,length(artifact))
 end
 allok||error("layout/modularoracle/input/guard check failed");println("PASS")
end
