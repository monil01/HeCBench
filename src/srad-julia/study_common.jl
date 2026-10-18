using CUDA,Printf
include("srad_device.jl")
function guarded(::Type{T},values) where T
 host=vcat(fill(T(16384),8),T.(values),fill(T(16384),8));base=CuArray(host)
 (host=host,base=base,data=view(base,9:length(host)-8),n=length(values))
end
function srad_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==69310&&pre[1]==4&&length(gold)==746001
 pos=2;cases=Any[]
 for ci in 1:4
  nr,nc,steps=Int.(pre[pos:pos+2]);lambda=Float32(pre[pos+3]);pos+=4;N=nr*nc;raw=pre[pos:pos+N-1];pos+=N;I=guarded(Float32,raw);other=[guarded(Float32,zeros(Float32,N)) for _ in 1:7];maps=Any[]
  for n in [nr,nr,nc,nc];push!(maps,guarded(Int32,pre[pos:pos+n-1]));pos+=n;end
  s,s2,dN,dS,dE,dW,c=other;iN,iS,jE,jW=maps
  push!(cases,(nr=Int32(nr),nc=Int32(nc),N=Int64(N),steps=steps,lambda=lambda,blocks=cld(N,256),I=I,s=s,s2=s2,dN=dN,dS=dS,dE=dE,dW=dW,c=c,iN=iN,iS=iS,jE=jE,jW=jW,hosttotal=zeros(Float32,1),hosttotal2=zeros(Float32,1),stats=zeros(Float32,5*steps)))
 end
 @assert pos==length(pre)+1
 function execute(c)
  arrays=(c.I,c.s,c.s2,c.dN,c.dS,c.dE,c.dW,c.c,c.iN,c.iS,c.jE,c.jW)
  for a in arrays[1:8];copyto!(a.base,a.host);end
  fill!(c.stats,0);raw=Array(c.I.base);ok=raw==c.I.host;CUDA.synchronize()
  start=time_ns();launch.extract(c);CUDA.synchronize();extract_seconds=(time_ns()-start)*1e-9;extracted=Array(c.I.base)
  start=time_ns()
  for it in 1:c.steps
   launch.prepare(c);no=Int32(c.N);mul=Int32(1);blocks=c.blocks
   while blocks!=0
    launch.reduce(c,no,mul,blocks);no=Int32(blocks)
    if blocks==1;blocks=0;else;mul*=Int32(256);blocks=cld(blocks,256);end
   end
   copyto!(c.hosttotal,1,c.s.data,1,1);copyto!(c.hosttotal2,1,c.s2.data,1,1)
   total=c.hosttotal[1];total2=c.hosttotal2[1];mean=total/Float32(c.N);mean2=mean*mean;var=total2/Float32(c.N)-mean2;q0=var/mean2
   c.stats[5*it-4]=total;c.stats[5*it-3]=total2;c.stats[5*it-2]=mean;c.stats[5*it-1]=var;c.stats[5*it]=q0
   launch.coefficient(c,q0);launch.update(c)
  end
  CUDA.synchronize();seconds=(time_ns()-start)*1e-9;before=Array(c.I.base)
  start=time_ns();launch.compress(c);CUDA.synchronize();compress_seconds=(time_ns()-start)*1e-9
  actual=map(a->Array(a.base),arrays);vals=Float64.(actual[1][9:end-8]);append!(vals,before[9:end-8])
  for a in actual[2:8];append!(vals,a[9:end-8]);end
  append!(vals,raw[9:end-8]);append!(vals,extracted[9:end-8])
  for (a,expected) in zip(actual[9:12],arrays[9:12]);ok &=a==expected.host;append!(vals,a[9:end-8]);end
  append!(vals,c.stats)
  for a in actual;g=vcat(a[1:8],a[end-7:end]);ok &=all(==(16384),g);append!(vals,g);end
  initial=Float64.(extracted[9:end-8]);final=Float64.(before[9:end-8]);coeff=actual[8][9:end-8];lo,hi=extrema(initial);ok &=all(isfinite,coeff)&&all(x->0<=x<=1,coeff)&&all(x->lo-1e-5<=x<=hi+1e-5,final)&&abs(sum(final)-sum(initial))<=3e-6*abs(sum(initial))&&all(>(0),c.stats[4:5:end])&&all(>(0),c.stats[5:5:end])
  (vals=vals,seconds=seconds,extract_seconds=extract_seconds,compress_seconds=compress_seconds,ok=ok)
 end
 for c in cases;@assert execute(c).ok;end
 for trial in 1:5
  art=copy(pre);times=Float64[];ext=Float64[];comp=Float64[];ok=true
  for c in cases;z=execute(c);append!(art,z.vals);push!(times,z.seconds);push!(ext,z.extract_seconds);push!(comp,z.compress_seconds);ok &=z.ok;end
  actual=art[length(pre)+1:end];@assert length(actual)==length(gold);errors=abs.(actual.-gold);ok &=all(isfinite,actual)&&all(errors.<=1e-4.+1e-5.*abs.(gold));ma=maximum(errors)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_SRAD_compute_prepare_reduce_D2Hstats_diffusion_iterations\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"extract_times_seconds_outside_primary_ROI\":%s,\"compress_times_seconds_outside_primary_ROI\":%s,\"compute_launches\":51,\"statistics_telemetry_F32_stores_inside_ROI\":50}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times),string(ext),string(comp))
 end
end
