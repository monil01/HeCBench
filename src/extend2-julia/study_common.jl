function study(allocate,launch)
 f=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixtures_i32.bin"))));oracle=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));states=[];i=1
 function buffer(v,s)
  h=vcat(fill(s,4),v,fill(s,4));a=allocate(eltype(v),length(h));copyto!(a,h);return(a=a,v=view(a,5:length(a)-4),h=h,s=s)
 end
 while i<=length(f)
  params=Tuple(f[i:i+10]);i+=11;qlen,tlen=Int.(params[1:2]);q=buffer(UInt8.(f[i:i+qlen-1]),UInt8(0xa5));i+=qlen;t=buffer(UInt8.(f[i:i+tlen-1]),UInt8(0xa5));i+=tlen;mat=buffer(Int8.(f[i:i+24]),Int8(-91));i+=25
  hh=buffer(zeros(Int32,qlen+1),Int32(-819225));ee=buffer(zeros(Int32,qlen+1),Int32(-819225));qp=buffer(zeros(Int8,qlen*5),Int8(-91));out=buffer(zeros(Int32,6),Int32(-819225));push!(states,(params,q,t,mat,hh,ee,qp,out))
 end
 function run(c)
  params,q,t,mat,hh,ee,qp,out=c;launch(q.v,t.v,mat.v,hh.v,ee.v,qp.v,out.v,params);CUDA.synchronize(CUDA.context())
 end
 for c in states;run(c);end
 for trial in 1:5
  payload=copy(f);elapsed=0.;ok=true;oi=1
  for c in states
   params,q,t,mat,hh,ee,qp,out=c
   for a in(hh,ee,qp,out);copyto!(a.a,a.h);end
   CUDA.synchronize(CUDA.context());start=time_ns();run(c);elapsed+=(time_ns()-start)*1e-9
   raw=[Array(a.a) for a in(q,t,mat,hh,ee,qp,out)]
   got=vcat(raw[7][5:end-4],raw[4][5:end-4],raw[5][5:end-4],Int32.(raw[6][5:end-4]));ok&=got==oracle[oi:oi+length(got)-1];oi+=length(got)
   for (j,a) in enumerate((q,t,mat,hh,ee,qp,out))
    r=raw[j];ok&=all(x==a.s for x in r[1:4])&&all(x==a.s for x in r[end-3:end]);if j<=3;ok&=r==a.h;end;append!(payload,Int32.(r))
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":0,\"timing_region\":\"sum22_preallocated_serial_extension_kernels_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullExtensionResultsScratchReadonlyAndGuards")
 end
end
