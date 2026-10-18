const SENT=-8192.25f0
function study(allocate,factory)
 f=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=1;zero_symbol=zeros(Float32,12288)
 function buffer(v)
  h=vcat(fill(SENT,4),v,fill(SENT,4));a=allocate(Float32,length(h));copyto!(a,h);return(a=a,v=view(a,5:length(a)-4),h=h)
 end
 while off<=length(f)
  ns,nv=Int.(f[off:off+1]);off+=3;rmu=buffer(f[off:off+nv-1]);off+=nv;imu=buffer(f[off:off+nv-1]);off+=nv;khost=copy(f[off:off+3nv-1]);off+=3nv;buf=[]
  for j in 1:5;push!(buf,buffer(f[off:off+ns-1]));off+=ns;end
  rout,iout,x,y,z=buf;counts=[min(4096,nv-j+1) for j in 1:4096:nv];rv=[view(rmu.v,j:j+c-1) for (j,c) in zip(1:4096:nv,counts)];iv=[view(imu.v,j:j+c-1) for (j,c) in zip(1:4096:nv,counts)];symbol,launch=factory(rv,iv,rout.v,iout.v,x.v,y.v,z.v,ns,counts);expected=zeros(Float32,12288)
  for (j,c) in enumerate(counts);copyto!(expected,1,khost,(j-1)*12288+1,3c);end
  push!(states,(ns,nv,rmu,imu,rout,iout,x,y,z,khost,counts,symbol,launch,expected))
 end
 function reset(c)
  ns,nv,rmu,imu,rout,iout,x,y,z,khost,counts,symbol,launch,expected=c;copyto!(rout.a,rout.h);copyto!(iout.a,iout.h);upload_constant(symbol,zero_symbol,1,12288);CUDA.synchronize(CUDA.context())
 end
 function run(c)
  ns,nv,rmu,imu,rout,iout,x,y,z,khost,counts,symbol,launch,expected=c
  for (j,count) in enumerate(counts);upload_constant(symbol,khost,(j-1)*12288+1,3count);launch(j);end
  CUDA.synchronize(CUDA.context())
 end
 for c in states;reset(c);run(c);end
 for trial in 1:5
  payload=copy(f);ok=true;elapsed=0.;err=0.;oi=1
  for c in states
   ns,nv,rmu,imu,rout,iout,x,y,z,khost,counts,symbol,launch,expected=c;reset(c);start=time_ns();run(c);elapsed+=(time_ns()-start)*1e-9
   r=Array(rout.v);im=Array(iout.v);got=vcat(r,im);ref=oracle[oi:oi+length(got)-1];oi+=length(got);e=abs.(Float64.(got).-ref);err=max(err,maximum(e));ok&=all(e.<=1e-4.+2e-5.*abs.(ref))
   for (j,b) in enumerate((rmu,imu,rout,iout,x,y,z))
    h=Array(b.a);ok&=all(v==SENT for v in h[1:4])&&all(v==SENT for v in h[end-3:end]);if j in(1,2,5,6,7);ok&=reinterpret(UInt32,h)==reinterpret(UInt32,b.h);end;append!(payload,h)
   end
   actual=read_constant(symbol);ok&=reinterpret(UInt32,actual)==reinterpret(UInt32,expected);append!(payload,actual)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",err,",\"timing_region\":\"sum3_constant_chunk_upload_and_Fourier_kernel_pipeline_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullFourierOracleActualConstantTailAndGuards")
 end
end
