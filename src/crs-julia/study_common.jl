const SENT=UInt64(0xdecafbad12345678);const BSENT=UInt32(0xa5a5a5a5)
function study(allocate,launch)
 fixture=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixtures_u64.bin"))));oracle=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"oracle_u64.bin"))));states=[];off=0
 while off<length(fixture)
  k,m,w,size,tasks,mode=Int.(fixture[off+1:off+6]);bi=UInt32.(fixture[off+7:off+6+k*w]);h=fixture[off+7+k*w:off+6+k*w+k*size*tasks];ah=vcat(SENT,h,SENT);bh=vcat(BSENT,bi,BSENT);a=allocate(UInt64,length(ah));b=allocate(UInt64,m*size*tasks+2);bm=allocate(UInt32,length(bh));copyto!(a,ah);copyto!(bm,bh);blocks=cld(size,4*div(32,w)*w);push!(states,(k,m,w,size,tasks,mode,a,b,bm,ah,bh,blocks));off+=6+k*w+k*size*tasks
 end
 function run(c)
  k,m,w,size,tasks,mode,a,b,bm,ah,bh,blocks=c;fill!(b,SENT);CUDA.synchronize(CUDA.context());start=time_ns()
  for task in 0:tasks-1
   launch(Int32(k),Int32(m),Int32(w),view(a,2+task*k*size:1+(task+1)*k*size),view(b,2+task*m*size:1+(task+1)*m*size),view(bm,2:k*w+1),Int32(size),blocks)
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9;got=Array(b);inback=Array(a);bmback=Array(bm);ok=inback==ah&&bmback==bh&&got[1]==SENT&&got[end]==SENT;return got[2:end-1],UInt64[inback[1],inback[end],got[1],got[end]],elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true
  for c in states
   vals,guards,t,pass=run(c);ok&=pass&&vals==oracle[oi+1:oi+length(vals)];oi+=length(vals);append!(out,vals);append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum100_preallocated_CauchyRS_kernel_pipelines_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("GFPolynomialParityAllWordsSourceMatrixAndGuards")
 end
end
