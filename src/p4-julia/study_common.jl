using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture_bits.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_full_outputs.bin"))))
const ST_FX,ST_FY,ST_NA,ST_NC,ST_NB=Int32.(ST_INPUT[1:5])
const ST_K=Int(ST_FX*ST_FY*ST_NA)
const ST_RANGES=Tuple(ST_INPUT[6:9]);const ST_OFFSET=ST_INPUT[10]
const ST_ANCHORS=ST_INPUT[11:34];const ST_BOTTOM=ST_INPUT[35:37];const ST_THRESHOLDS=ST_INPUT[38:40]
const ST_CLS=ST_INPUT[41:40+ST_K*3]
const ST_BLEN=ST_K*7+17;const ST_OLEN=ST_K*9+17;const ST_SLEN=ST_K+17
const ST_BOX=ST_INPUT[41+ST_K*3:40+ST_K*3+ST_BLEN]
const ST_DIR=ST_INPUT[41+ST_K*3+ST_BLEN:end]
const ST_LEN=1+ST_BLEN+ST_OLEN+ST_SLEN
const ST_GUARD=-12345f0
function st_prepare()
 (CuArray(ST_CLS),CuArray(ST_BOX),CuArray(ST_DIR),CuArray(ST_ANCHORS),CuArray(ST_BOTTOM),CUDA.fill(ST_GUARD,ST_OLEN),CUDA.fill(ST_GUARD,ST_SLEN),CUDA.zeros(Int32,1))
end
function st_reset!(d)
 copyto!(d[2],ST_BOX);CUDA.fill!(d[6],ST_GUARD);CUDA.fill!(d[7],ST_GUARD);CUDA.fill!(d[8],Int32(0))
end
function canonical_payload(d)
 n=Int(Array(d[8])[1]);@assert 0<=n<=ST_K
 changed,bnd,score=Array(d[2]),Array(d[6]),Array(d[7]);raw=vcat(Float32[n],changed,bnd,score)
 order=sortperm(Float32[bnd[(i-1)*9+9] for i in 1:n]);canonical=copy(raw);bs=2+ST_BLEN;ss=bs+ST_OLEN
 for j in 1:n
  k=order[j]
  canonical[bs+(j-1)*9:bs+j*9-1]=bnd[(k-1)*9+1:k*9]
  canonical[ss+j-1]=score[k]
 end
 canonical,raw,n
end
function check_payload(c,g)
 n=Int(g[1]);bs=2+ST_BLEN;ss=bs+ST_OLEN;accept=falses(ST_K)
 for j in 1:n;accept[Int(g[bs+(j-1)*9+8])÷7+1]=true;end
 ok=c[1]==g[1];ma=0.;mr=0.;scoreerr=0.
 for i in eachindex(c)
  x=Float64(c[i]);gold=g[i];err=abs(x-gold);ma=max(ma,err);mr=max(mr,err/max(abs(gold),1e-12))
  strict=i==1||gold==Float64(ST_GUARD)
  if 2<=i<2+ST_K*7;strict|=!accept[(i-2)÷7+1];end
  if bs<=i<ss;strict|=(i-bs)%9>=7;end
  tol=i>=ss ? 1e-6 : 1e-3
  if i>=ss&&gold!=Float64(ST_GUARD);scoreerr=max(scoreerr,err);end
  ok&=isfinite(x)&&(strict ? x==gold : err<=tol)
 end
 ok,ma,mr,scoreerr
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);d=st_prepare()
 for thresh in ST_THRESHOLDS;st_reset!(d);CUDA.synchronize();launch!(d,thresh);CUDA.synchronize();end
 side=joinpath("/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/p4",model*"_"*string(getpid())*"_"*string(time_ns()));mkpath(side)
 for trial in 1:5
  times=Float64[];raw=Float32[];got=Float32[];counts=Int[];ok=true;ma=0.;mr=0.;scoreerr=0.;gct=0.;gcp=0
  for (case,thresh) in enumerate(ST_THRESHOLDS)
   elapsed=0.
   for _ in 1:rep
    st_reset!(d);CUDA.synchronize();gb=Base.gc_num();st=time_ns();launch!(d,thresh);CUDA.synchronize();elapsed+=(time_ns()-st)*1e-9;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   end
   c,rr,n=canonical_payload(d);good,ea,er,es=check_payload(c,@view ST_GOLD[(case-1)*ST_LEN+1:case*ST_LEN]);ok&=good;ma=max(ma,ea);mr=max(mr,er);scoreerr=max(scoreerr,es);append!(got,c);append!(raw,rr);push!(counts,n);push!(times,elapsed/rep)
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got));sidefile=joinpath(side,"raw_trial$(trial).bin");write(sidefile,raw)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"max_score_abs_error\":%.17g,\"timing_region\":\"sum3_independent_postprocess_kernels_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"accepted_counts\":[%s],\"raw_output_sidecar\":\"%s\",\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,scoreerr,rep,join(times,","),join(counts,","),sidefile,gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
