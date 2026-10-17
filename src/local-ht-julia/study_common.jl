include(joinpath(@__DIR__,"study_fixture.jl"))
function canonical_records(keylen,keypos,counts,exts,extcounts,visited_len,visited_pos,visited_val,longest,temporary,walklens,hashes)
 kl,kp,co,ex,ec,vl,vp,vv,lo,te,wl,ha=Array.((keylen,keypos,counts,exts,extcounts,visited_len,visited_pos,visited_val,longest,temporary,walklens,hashes))
 output=UInt32[]
 for group in 0:4
  push!(output,wl[group+1]);append!(output,UInt32.(lo[group*16+1:(group+1)*16]));append!(output,UInt32.(te[group*137+1:(group+1)*137]))
  records=Vector{Vector{UInt32}}()
  for slot in group*512+1:(group+1)*512
   kl[slot]==EMPTY_KEY&&continue
   kl[slot]==21||Base.error("Invalid key length in occupied table")
   pos=Int(kp[slot]);key=UInt32.(FIX_reads[pos+1:pos+21]);record=vcat(key,co[(slot-1)*8+1:slot*8],UInt32[ex[slot],ec[slot],ha[slot]]);push!(records,record)
  end
  sort!(records;by=record->Tuple(record[1:21]));length(records)<=64||Base.error("Canonical key capacity exceeded")
  push!(output,length(records));for record in records;append!(output,record);end;append!(output,zeros(UInt32,(64-length(records))*32))
  seen=Vector{Vector{UInt32}}()
  for slot in group*16+1:(group+1)*16
   vl[slot]==EMPTY_KEY&&continue
   vl[slot]==21||Base.error("Invalid visited key length")
   pos=Int(vp[slot]);push!(seen,vcat(UInt32.(te[pos+1:pos+21]),UInt32[vv[slot]]))
  end
  sort!(seen;by=record->Tuple(record[1:21]));push!(output,length(seen));for record in seen;append!(output,record);end;append!(output,zeros(UInt32,(16-length(seen))*22))
 end
 output
end
function study(run_main!,run_hash!)
 inputs=(CuArray(FIX_ctg_offsets),CuArray(FIX_contigs),CuArray(FIX_reads),CuArray(FIX_quals),CuArray(FIX_read_offsets),CuArray(FIX_read_counts),CuArray(FIX_prefix_ht),CuArray(fill(10.,5)))
 work=(CUDA.zeros(UInt32,2560),CUDA.zeros(Int32,2560),CUDA.zeros(UInt32,2560*8),CUDA.zeros(UInt8,2560),CUDA.zeros(UInt32,2560),CUDA.zeros(UInt32,80),CUDA.zeros(Int32,80),CUDA.zeros(UInt8,80),CUDA.zeros(UInt8,80),CUDA.zeros(UInt8,685),CUDA.zeros(UInt32,5),CUDA.zeros(UInt32,2560))
 want=zeros(UInt32,EXPECTED_COUNT)
 for (i,value) in EXPECTED_NONZERO;want[i]=value;end
 function trial!()
  for array in work;CUDA.fill!(array,0);end
  CUDA.synchronize();t=time_ns()
  run_main!(inputs...,work[1:11]...,Int32(21),Int32(16),Int32(121),Int32(5),Int32(33))
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9
  run_hash!(work[1],work[2],inputs[3],work[12],Int32(512),Int32(2560));CUDA.synchronize()
  got=canonical_records(work...);return got,elapsed,got==want
 end
 trial!()
 for trial in 1:5
  got,elapsed,ok=trial!();output=vcat(FIX_input_prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  if !ok
   mismatches=findall(got.!=want);println(stderr,"Local assembly mismatch slots ",first(mismatches,min(30,length(mismatches)))," total ",length(mismatches))
  end
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":15991,\"max_abs_error\":0,\"timing_region\":\"integrated_right_kmer_count_quality_walk_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":15991,\"byte_order\":\"little\"}}")
  ok||Base.error("Independent local assembly dictionary/walk checks failed")
 end
end
