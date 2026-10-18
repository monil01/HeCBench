
const BASE=dirname(@__FILE__);const G=16;const IS=Int32(0x5a5a5a5a);const VS=UInt64(0x12345678fedcba98)
readv(::Type{T},name) where T = collect(reinterpret(T,read(joinpath(BASE,name))))
function appendi!(dest,a)
 append!(dest,reinterpret(UInt32,a))
end
function appendv!(dest,a)
 append!(dest,reinterpret(UInt32,a))
end
function run_pipeline(launch,c)
 A,B,V,W,offsets,ids,dc,hc,stream,event,keys,vals,oracle,oraclev,off,counts,mode=c
 ns=length(off)-1;beg=offsets;ends=view(offsets,2:length(offsets));alln=Int32(ns)
 CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(dc)),UInt8(0),14*4;stream=stream)
 launch(Val(:bb_bin_histo_4),dc,beg,ends,alln;blocks=cld(ns,256),threads=256)
 GC.@preserve hc unsafe_copyto!(pointer(hc),pointer(dc),14;async=true,stream=stream)
 CUDA.record(event,stream)
 launch(Val(:bb_bin_group_5),ids,dc,beg,ends,alln;blocks=cld(ns,256),threads=256)
 CUDA.synchronize(event)
 maxsize=hc[14];hc[14]=Int32(ns)
 count=Int(hc[2]-hc[1])
 if count>0
  bin=view(ids,Int(hc[1])+1:length(ids));blocks=cld(count,256)
  if mode==0
   launch(Val(:gen_copy_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  else
   launch(Val(:gen_copy_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  end
 end
 count=Int(hc[3]-hc[2])
 if count>0
  bin=view(ids,Int(hc[2])+1:length(ids));blocks=cld(count,128)
  if mode==0
   launch(Val(:gen_bk256_wp2_tc1_r2_r2_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  else
   launch(Val(:gen_bk256_wp2_tc1_r2_r2_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  end
 end
 count=Int(hc[4]-hc[3])
 if count>0
  bin=view(ids,Int(hc[3])+1:length(ids));blocks=cld(count,64)
  if mode==0
   launch(Val(:gen_bk128_wp2_tc2_r3_r4_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_wp2_tc2_r3_r4_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[5]-hc[4])
 if count>0
  bin=view(ids,Int(hc[4])+1:length(ids));blocks=cld(count,64)
  if mode==0
   launch(Val(:gen_bk128_wp2_tc4_r5_r8_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_wp2_tc4_r5_r8_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[6]-hc[5])
 if count>0
  bin=view(ids,Int(hc[5])+1:length(ids));blocks=cld(count,32)
  if mode==0
   launch(Val(:gen_bk128_wp4_tc4_r9_r16_strd_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_wp4_tc4_r9_r16_strd_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[7]-hc[6])
 if count>0
  bin=view(ids,Int(hc[6])+1:length(ids));blocks=cld(count,16)
  if mode==0
   launch(Val(:gen_bk128_wp8_tc4_r17_r32_strd_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_wp8_tc4_r17_r32_strd_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[8]-hc[7])
 if count>0
  bin=view(ids,Int(hc[7])+1:length(ids));blocks=cld(count,8)
  if mode==0
   launch(Val(:gen_bk128_wp16_tc4_r33_r64_strd_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_wp16_tc4_r33_r64_strd_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[9]-hc[8])
 if count>0
  bin=view(ids,Int(hc[8])+1:length(ids));blocks=cld(count,32)
  if mode==0
   launch(Val(:gen_bk256_wp8_tc16_r65_r128_strd_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  else
   launch(Val(:gen_bk256_wp8_tc16_r65_r128_strd_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  end
 end
 count=Int(hc[10]-hc[9])
 if count>0
  bin=view(ids,Int(hc[9])+1:length(ids));blocks=cld(count,8)
  if mode==0
   launch(Val(:gen_bk256_wp32_tc8_r129_r256_strd_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  else
   launch(Val(:gen_bk256_wp32_tc8_r129_r256_strd_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  end
 end
 count=Int(hc[11]-hc[10])
 if count>0
  bin=view(ids,Int(hc[10])+1:length(ids));blocks=cld(count,1)
  if mode==0
   launch(Val(:gen_bk128_tc4_r257_r512_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  else
   launch(Val(:gen_bk128_tc4_r257_r512_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=128)
  end
 end
 count=Int(hc[12]-hc[11])
 if count>0
  bin=view(ids,Int(hc[11])+1:length(ids));blocks=cld(count,1)
  if mode==0
   launch(Val(:gen_bk256_tc4_r513_r1024_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  else
   launch(Val(:gen_bk256_tc4_r513_r1024_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=256)
  end
 end
 count=Int(hc[13]-hc[12])
 if count>0
  bin=view(ids,Int(hc[12])+1:length(ids));blocks=cld(count,1)
  if mode==0
   launch(Val(:gen_bk512_tc4_r1025_r2048_orig_6),A,B,beg,ends,bin,Int32(count);blocks=blocks,threads=512)
  else
   launch(Val(:gen_bk512_tc4_r1025_r2048_orig_8),A,V,B,W,beg,ends,bin,Int32(count);blocks=blocks,threads=512)
  end
 end
 nlong=Int(hc[14]-hc[13])
 if nlong>0
  bin=view(ids,Int(hc[13])+1:length(ids));grid=(nlong,cld(Int(maxsize),2048))
  if mode==0
   launch(Val(:kern_block_sort_6),A,B,beg,ends,bin,Int32(2048);blocks=grid,threads=512)
   src,dst=B,A;swaps=1
   stride=2048
   while stride<maxsize
    launch(Val(:kern_block_merge_7),src,dst,beg,ends,bin,Int32(stride),Int32(2048);blocks=grid,threads=128)
    src,dst=dst,src;swaps+=1;stride*=2
   end
   if isodd(swaps);src,dst=dst,src;end
   launch(Val(:kern_copy_6),src,dst,beg,ends,bin,Int32(2048);blocks=grid,threads=128)
  else
   launch(Val(:kern_block_sort_8),A,V,B,W,beg,ends,bin,Int32(2048);blocks=grid,threads=512)
   src,dst=B,A;sv,dv=W,V;swaps=1;stride=2048
   while stride<maxsize
    launch(Val(:kern_block_merge_9),src,sv,dst,dv,beg,ends,bin,Int32(stride),Int32(2048);blocks=grid,threads=128)
    src,dst=dst,src;sv,dv=dv,sv;swaps+=1;stride*=2
   end
   if isodd(swaps);src,dst=dst,src;sv,dv=dv,sv;end
   launch(Val(:kern_copy_8),src,sv,dst,dv,beg,ends,bin,Int32(2048);blocks=grid,threads=128)
  end
 end
 nothing
end
function segsort_study(launch)
 n,ns=parse.(Int,split(read(joinpath(BASE,"study_sizes.txt"),String)));off=readv(Int32,"offsets_i32.bin");length(off)==ns+1||error("offsetcount")
 counts=zeros(Int32,13)
 for i=1:ns
  sz=off[i+1]-off[i];bin=0;while bin<12&&sz>1<<bin;bin+=1;end;counts[bin+1]+=1
 end
 cases=map(0:1)do mode
  keys=readv(Int32,"keys_case$(mode+1)_i32.bin");oracle=readv(Int32,"oracle_case$(mode+1)_i32.bin");vals=mode==1 ? readv(UInt64,"values_case2_u64.bin") : UInt64[];oraclev=mode==1 ? readv(UInt64,"oracle_values_case2_u64.bin") : UInt64[]
  A=CuArray(fill(IS,n+G));B=CuArray(fill(IS,n+G));V=CuArray(fill(VS,mode==1 ? n+G : G));W=CuArray(fill(VS,mode==1 ? n+G : G));offsets=CuArray(vcat(off,fill(IS,G)));ids=CuArray(fill(IS,ns+G));dc=CuArray(fill(IS,14+G));hc=fill(IS,14+G);CUDA.pin(hc)
  stream=CUDA.CuStream();event=CUDA.CuEvent()
  c=(A,B,V,W,offsets,ids,dc,hc,stream,event,keys,vals,oracle,oraclev,off,counts,mode)
  CUDA.stream!(stream)do
   copyto!(A,1,keys,1,n);if mode==1;copyto!(V,1,vals,1,n);end;CUDA.synchronize(stream)
   run_pipeline(launch,c);CUDA.synchronize(stream)
  end
  c
 end
 for trial=1:5
  prefix=UInt32[];values=UInt32[];guards=UInt32[];times=Float64[];rawids=Vector{Int32}[];ok=true;mismatches=0;idchecks=0
  for c in cases
   A,B,V,W,offsets,ids,dc,hc,stream,event,keys,vals,oracle,oraclev,off,counts,mode=c
   CUDA.stream!(stream)do
    copyto!(A,1,keys,1,n);if mode==1;copyto!(V,1,vals,1,n);end;CUDA.synchronize(stream)
   end
   appendi!(prefix,Array(A)[1:n]);if mode==1;appendv!(prefix,Array(V)[1:n]);end;appendi!(prefix,Array(offsets)[1:ns+1]);append!(prefix,UInt32.([n,ns,mode,13,2048,5000,1,0]))
   elapsed=CUDA.stream!(stream)do
    CUDA.synchronize(stream);start=time_ns();run_pipeline(launch,c);CUDA.synchronize(stream);(time_ns()-start)*1e-9
   end
   push!(times,elapsed);out=Array(B)[1:n];mismatches+=sum(out.!=oracle);ok &= out==oracle;appendi!(values,out)
   if mode==1;v=Array(W)[1:n];mismatches+=sum(v.!=oraclev);ok &= v==oraclev;appendv!(values,v);end
   counters=Array(dc)[1:14];id=Array(ids)[1:ns];seen=zeros(Int32,ns);total=0
   for bin=0:12
    ok &= hc[bin+1]==total
    for pos=total+1:total+counts[bin+1]
     seg=Int(id[pos]);pass=0<=seg<ns
     if pass
      seen[seg+1]+=1;sz=off[seg+2]-off[seg+1];b=0;while b<12&&sz>1<<b;b+=1;end;pass=b==bin
     end
     ok &= pass;idchecks+=1
    end
    total+=counts[bin+1];ok &= counters[bin+1]==total
   end
   ok &= all(==(1),seen)&&counters[14]==5000&&hc[14]==ns
   for i=1:14;push!(values,UInt32(counters[i]));push!(values,UInt32(hc[i]));end
   for (arr,len)in ((A,n),(B,n),(offsets,ns+1),(ids,ns),(dc,14))
    g=Array(arr)[len+1:end];ok &= all(==(IS),g);appendi!(guards,g)
   end
   hg=hc[15:end];ok &= all(==(IS),hg);appendi!(guards,hg)
   if mode==1;for arr in (V,W);g=Array(arr)[n+1:end];ok &= all(==(VS),g);appendv!(guards,g);end;end
   push!(rawids,id)
  end
  artifact_values=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,artifact_values))
  idsjson="["*join(("["*join(x,",")*"]" for x in rawids),",")*"]"
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"mismatch_output_values\":%d,\"error_metric_scope\":\"Computedexactkey/value/countermismatchflag; relativeerror notapplicable\",\"timing_region\":\"sum_keys_and_pairs_full_bb_bin_dispatch_network_large_merge_stream_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"internal_bin_ids_checked\":%d,\"internal_bin_ids_raw_not_public_output\":%s,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),mismatches>0 ? 1 : 0,mismatches,idchecks,idsjson,join(times,","),path,length(artifact_values));flush(stdout);ok||error("Fullsegmentedkeys/pairs/counters/binpermutation/guardsfailed")
 end
end
