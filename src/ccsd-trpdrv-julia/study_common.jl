using CUDA,Printf
include(joinpath(@__DIR__,"ccsd_math.jl"))
const CV=33;const CO=3
function study_main(launch!)
 repeat=isempty(ARGS) ? 10 : parse(Int,ARGS[1]);nv=CV
 eo=vcat([-1.0,-1.25,-1.5],[0.25+((j-1)%7)/32 for j in 1:nv]);base=[Float64[((j-1)*17+(a-1)*11)%67-33 for j in 1:(a<=8 ? nv^2 : nv)]./128 for a in 1:14];fixtures=[deepcopy(base) for _ in 1:3]
 for a in 1:8;fixtures[2][a].*=2;end
 fixtures[3][11].=0;fixtures[3][14].=0
 triples=[(i,j,k) for j in 1:CO for i in 1:CO for k in 1:i];eas=[eo[1]-(eo[i]+eo[j]+eo[k]) for (i,j,k) in triples];nt=length(triples)
 gold=zeros(BigFloat,4,nt,3)
 setprecision(256) do
  bigs=[map(x->BigFloat.(x),z) for z in fixtures];be=BigFloat.(eo)
  for f in 1:3,t in 1:nt,c in 1:nv,b in 1:nv
   terms=ccsd_terms(b,c,bigs[f]...,be,BigFloat(eas[t]),0,CO,nv)
   for q in 1:4;gold[q,t,f]+=terms[q];end
  end
 end
 # These identities check polynomial structure independently of summation precision.
 @assert maximum(abs.(gold[[1,3],:,2].-4gold[[1,3],:,1]))<big"1e-60"
 @assert maximum(abs.(gold[[2,4],:,2].-2gold[[2,4],:,1]))<big"1e-60"
 @assert all(iszero,gold[[2,4],:,3])
 da=[map(CuArray,x) for x in fixtures];de=CuArray(eo);outs=[CUDA.zeros(Float64,4) for f in 1:3,t in 1:nt]
 for f in 1:3,t in 1:nt;launch!(da[f],de,eas[t],outs[f,t],nv);end;CUDA.synchronize()
 prefix=vcat(eo,[vcat(x...) for x in fixtures]...)
 for trial in 1:5
  for o in outs;fill!(o,0.0);end;CUDA.synchronize();start=time_ns()
  for _ in 1:repeat,f in 1:3,t in 1:nt;launch!(da[f],de,eas[t],outs[f,t],nv);end
  CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/repeat;got=Float64[];err=0.0
  for f in 1:3
   emp4=0.0;emp5=0.0
   for t in 1:nt
    v=Array(outs[f,t])./repeat;append!(got,v);err=max(err,maximum(abs.(v.-Float64.(gold[:,t,f]))));i,j,k=triples[t];emp4+=v[1];emp5+=v[2];if i!=k;emp4+=v[3];emp5+=v[4];end
   end
   append!(got,[emp4,emp5])
  end
  ok=all(isfinite,got)&&err<1e-10;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");artifact=vcat(prefix,got);write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"54_complete_CCSD_energy_kernels\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,repeat,path,length(artifact));ok||error("CCSD high precision check failed")
 end
end
