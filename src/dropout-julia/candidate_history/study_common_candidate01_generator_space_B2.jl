const FSENT=-8192.25f0
const MSENT=UInt8(0xa5)
const GSENT=UInt32(0xa5a5a5a5)
const SSENT=UInt64(0xdecafbad12345678)
function study(allocate,launch,probe)
 fixture=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixtures_u32.bin"))));packed=read(joinpath(@__DIR__,"oracle_masks_packed.bin"));seqs=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"probe_subseq_u64.bin"))));expectedraw=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_rng_u32.bin"))));seed=UInt64(fixture[3])|(UInt64(fixture[4])<<32);offset=UInt64(fixture[5])|(UInt64(fixture[6])<<32);probs=collect(reinterpret(Float32,fixture[8:11]));states=[];off=11;oracle_base=0
 for ni in 1:2
  n=Int(fixture[off+1]);mode=Int(fixture[off+2]);input=collect(reinterpret(Float32,fixture[off+3:off+2+n]));off+=2+n
  for vec in (1,2,4)
   ah=vcat(fill(FSENT,4),input,fill(FSENT,4));a=allocate(Float32,n+8);b=allocate(Float32,n+8);mask=allocate(UInt8,n+32);copyto!(a,ah);av=view(a,5:n+4);bv=view(b,5:n+4);mv=view(mask,17:n+16)
   if vec>1
    VF=NTuple{vec,VecElement{Float32}};VM=NTuple{vec,VecElement{UInt8}};av=reinterpret(VF,av);bv=reinterpret(VF,bv);mv=reinterpret(VM,mv)
    @assert sizeof(VF)==4vec&&sizeof(VM)==vec&&UInt(pointer(av))%UInt(4vec)==0&&UInt(pointer(mv))%UInt(vec)==0
   end
   push!(states,(n,mode,vec,a,b,mask,av,bv,mv,ah,input,oracle_base));oracle_base+=4n
  end
 end
 sqh=vcat(SSENT,seqs,SSENT);sq=allocate(UInt64,7);rr=allocate(UInt32,82);copyto!(sq,sqh);fill!(rr,GSENT);probe(view(rr,2:81),view(sq,2:6),seed,offset);CUDA.synchronize(CUDA.context());rawprobe=Array(rr);seqback=Array(sq);rawprobe[2:81]==expectedraw&&rawprobe[1]==GSENT&&rawprobe[end]==GSENT&&seqback==sqh||error("PhiloxRaw80AndProbeGuards")
 probepayload=vcat(rawprobe,collect(reinterpret(UInt32,seqback[[1,7]])));prefix=vcat(fixture,collect(reinterpret(UInt32,seqs)))
 expectedmask(bit)=UInt8((packed[(bit>>3)+1]>>(bit&7))&UInt8(1))
 function capture(c,pi)
  n,mode,vec,a,b,mask,av,bv,mv,ah,input,base=c;got=Array(b);bits=collect(reinterpret(UInt32,got));mb=Array(mask);ab=Array(a);ok=reinterpret(UInt32,ab)==reinterpret(UInt32,ah)&&all(isequal(x,FSENT)for x in got[1:4])&&all(isequal(x,FSENT)for x in got[end-3:end])&&all(x==MSENT for x in mb[1:16])&&all(x==MSENT for x in mb[end-15:end]);scale=1f0/probs[pi];result=bits[5:n+4];m=mb[17:n+16]
  for i in 1:n
   keep=expectedmask(base+(pi-1)*n+i-1);expected=input[i]*Float32(keep)*scale;ok&=m[i]==keep&&result[i]==reinterpret(UInt32,expected)
  end
  guards=vcat(bits[1:4],bits[end-3:end],collect(reinterpret(UInt32,ab[1:4])),collect(reinterpret(UInt32,ab[end-3:end])),UInt32.(mb[1:16]),UInt32.(mb[end-15:end]));return vcat(result,UInt32.(m),guards),ok
 end
 function run(c,check)
  n,mode,vec,a,b,mask,av,bv,mv,ah,input,base=c;fill!(b,FSENT);fill!(mask,MSENT);CUDA.synchronize(CUDA.context());start=time_ns()
  for probability in probs;launch(vec,av,bv,mv,UInt32(n),probability,seed,offset);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9
  if !check;return elapsed,UInt32[],true;end
  # Save actualtimedp4 BEFORE replay overwrites the output; p1..3 are explicituntimed deterministicreplays.
  final,ok=capture(c,4);payload=UInt32[]
  for pi in 1:3
   launch(vec,av,bv,mv,UInt32(n),probs[pi],seed,offset);CUDA.synchronize(CUDA.context());piece,pass=capture(c,pi);append!(payload,piece);ok&=pass
  end
  append!(payload,final);return elapsed,payload,ok
 end
 for c in states;run(c,false);end
 for trial in 1:5
  out=vcat(prefix,probepayload);elapsed=0.;ok=true;times=Float64[]
  for c in states;t,payload,pass=run(c,true);elapsed+=t;push!(times,t);append!(out,payload);ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum6_original_four_probability_kernel_pipelines_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"probability_steps\":4,\"timed_launches\":24,\"untimed_probability_replay_launches\":18,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("PhiloxExactAllMasksScaledFloatBitsAndAlignedGuards")
 end
end
