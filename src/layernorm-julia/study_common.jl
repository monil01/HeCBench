using CUDA,Printf
function layernorm_study(factory)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==18505&&length(gold)==566136
 pos=2;cs=Any[];jobs=Any[]
 for ci in 1:6
  N,C,mode=Int32.(prefix[pos:pos+2]);pos+=3;hs=Any[];ds=Any[]
  for n in [N*C,C,C]
   h=vcat(fill(-12345f0,8),Float32.(prefix[pos:pos+n-1]),fill(-12345f0,8));pos+=n;push!(hs,h);push!(ds,CuArray(h))
  end
  outs=[CUDA.fill(-12345f0,n+16) for n in [N*C,N,N]];iv=[view(x,9:length(x)-8) for x in ds];ov=[view(x,9:length(x)-8) for x in outs];vec(v)=C%4==0 ? reinterpret(NTuple{4,VecElement{Float32}},v) : v
  args0=(ov[1],vec(ov[1]),ov[2],ov[3],iv[1],vec(iv[1]),iv[2],vec(iv[2]),iv[3],vec(iv[3]),N,C);argswarp=(ov[1],ov[2],ov[3],iv[1],iv[2],iv[3],N,C);c=(N=N,C=C,mode=mode,hs=hs,ds=ds,outs=outs,args0=args0,argswarp=argswarp);push!(cs,c)
  for k in 0:2,bt in (32,64,128,256,512,1024)
   launch=factory(c,k,bt);launch();CUDA.synchronize();push!(jobs,(c,launch))
  end
 end
 @assert pos==18505+1
 for trial in 1:5
  art=copy(prefix);times=Float64[];ok=true;ma=0.;mr=0.;go=0
  for (c,launch) in jobs
   for x in c.outs;fill!(x,-12345f0);end
   CUDA.synchronize();t=time_ns();for rep in 1:10;launch();CUDA.synchronize();end;push!(times,(time_ns()-t)*1e-9/10)
   for (k,x) in enumerate(c.outs)
    got=Array(x);ref=gold[go+1:go+length(got)];go+=length(got);ok&=all(isfinite,got)&&Float64.(got[1:8])==ref[1:8]&&Float64.(got[end-7:end])==ref[end-7:end];v=got[9:end-8];r=ref[9:end-8];er=abs.(Float64.(v).-r);ok&=all(er.<=1e-5 .+2e-6.*abs.(r));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(r).+1e-300)));if k==3;ok&=all(>(0),v);end
    if k==1
     for j in 0:length(v)-1
      if c.mode==1||(j%c.C)%19==0;ok&=v[j+1]==c.hs[3][9+j%c.C];end
     end
    end
    append!(art,Float64.(got))
   end
   for (h,x) in zip(c.hs,c.ds)
    got=Array(x);ref=gold[go+1:go+length(got)];go+=length(got);ok&=reinterpret(UInt8,got)==reinterpret(UInt8,h)&&Float64.(got)==ref;append!(art,Float64.(got))
   end
  end
  @assert go==566136&&length(art)==584641;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":566136,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum108_layernorm_3alg6width_kernel_sync_batches_per10\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":584641,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",ma,mr,file,string(times))
 end
end
