using CUDA,Printf
function kalman_study(launch!)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==59311&&length(gold)==506124
 pos=2;cs=Any[]
 for ci in 0:3
  ns,no,nf,inter,conf=Int32.(prefix[pos:pos+4]);pos+=5;lens=[ns*no,ns*64,ns*8,ns*64,ns*64,ns*8,ns];hs=Any[];ds=Any[]
  for n in lens
   h=vcat(fill(-12345.,8),prefix[pos:pos+n-1],fill(-12345.,8));pos+=n;push!(hs,h);push!(ds,CuArray(h))
  end
  sizes=[ns*no,ns*no,ns,ns*nf,ns*nf];outs=[CUDA.fill(-12345.,n+16) for n in sizes];iv=[view(x,9:length(x)-8) for x in ds];ov=[view(x,9:length(x)-8) for x in outs];push!(cs,(ns=ns,no=no,nf=nf,inter=inter!=0,conf=conf!=0,hs=hs,ds=ds,outs=outs,iv=iv,ov=ov))
 end
 @assert pos==59311+1
 args(c,nd)=(c.iv[1],c.no,c.iv[2],c.iv[3],c.iv[4],c.iv[5],c.iv[6],c.inter,c.iv[7],c.ns,c.ov[1],c.ov[2],c.ov[3],Int32(nd),c.nf,c.ov[4],c.conf,c.ov[5])
 for (ci,c) in enumerate(cs),nd in (ci==4 ? (0,7) : 0:7);launch!(c,args(c,nd));CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);times=Float64[];ok=true;ma=0.;mr=0.;go=0
  for (ci,c) in enumerate(cs),nd in (ci==4 ? (0,7) : 0:7)
   for x in c.outs;fill!(x,-12345.);end
   CUDA.synchronize();t=time_ns();launch!(c,args(c,nd));CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   for (k,x) in enumerate(c.outs)
    got=Array(x);ref=gold[go+1:go+length(got)];go+=length(got);ok&=all(isfinite,got)&&got[1:8]==ref[1:8]&&got[end-7:end]==ref[end-7:end]
    v=got[9:end-8];r=ref[9:end-8]
    if (k==5&&!c.conf)||(k==3&&nd>=c.no);ok&=v==r
    elseif !isempty(v)
     er=abs.(v.-r);ok&=all(er.<=1e-9 .+1e-9.*abs.(r));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(r).+1e-300)));if k==2||k==5;ok&=all(>(0),v);end
    end
    append!(art,got)
   end
   for (h,x) in zip(c.hs,c.ds)
    got=Array(x);ref=gold[go+1:go+length(got)];go+=length(got);ok&=reinterpret(UInt8,got)==reinterpret(UInt8,h)&&got==ref;append!(art,got)
   end
  end
  @assert go==506124&&length(art)==565435;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":506124,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum26_full_Kalman_filter_forecast_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":565435,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",ma,mr,file,string(times))
 end
end
