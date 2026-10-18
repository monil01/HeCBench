using CUDA,Printf
const CONV_N=2048
const CONV_GUARD=-12345
const CONV_BTS=(64,128,256,512,1024)
function study_decode(words,T)
 T===Float64 ? collect(reinterpret(Float64,words)) : T===Float32 ? collect(reinterpret(Float32,UInt32.(words))) : collect(reinterpret(Int16,UInt16.(words)))
end
function study_main(launch!)
 words=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==12432
 Ts=(Float64,Float32,Int16);inputs=Any[];hostinputs=Any[];dmask=Any[CUDA.zeros(T,10) for T in Ts];offset=0
 for fixture in 0:1,T in Ts
  a=study_decode(words[offset+1:offset+CONV_N],T);push!(hostinputs,a);push!(inputs,CuArray(a));offset+=CONV_N
 end
 groups=Any[]
 for fixture in 0:1,MW in (3,5,7,9),ti in 1:3
  T=Ts[ti];mask=study_decode(words[offset+1:offset+MW],T);offset+=MW;ic=fixture*3+ti;a=hostinputs[ic];gold=fill(Float64(CONV_GUARD),CONV_N+16,15)
  @assert maximum(abs.(Int64.(a)))*sum(abs,Int64.(mask))<typemax(Int16)
  for i0 in 0:CONV_N-1
   acc=Int64(0)
   for j in 0:MW-1;idx=i0-MW÷2+j;if 0<=idx<CONV_N;acc+=Int64(a[idx+1])*Int64(mask[j+1]);end;end
   for path in 1:15;gold[i0+9,path]=Float64(acc);end
  end
  out=CUDA.fill(T(CONV_GUARD),CONV_N+16,15);views=[view(out,9:CONV_N+8,path) for path in 1:15]
  push!(groups,(a=inputs[ic],mask=mask,dm=dmask[ti],mw=Int32(MW),out=out,views=views,gold=gold))
 end
 @assert offset==length(words)
 for g in groups
  copyto!(g.dm,1,g.mask,1,length(g.mask));CUDA.synchronize()
  for mode in 1:3,bi in 1:5;launch!(g.a,g.views[(mode-1)*5+bi],g.dm,g.mw,Val(mode),Val(CONV_BTS[bi]));end
  CUDA.synchronize()
 end
 for trial in 1:5
  elapsed=0.0;ok=true;err=0.0;art=copy(words)
  for g in groups
   copyto!(g.dm,1,g.mask,1,length(g.mask));fill!(g.out,eltype(g.a)(CONV_GUARD));CUDA.synchronize();@assert Array(g.dm)[1:length(g.mask)]==g.mask
   t=time_ns()
   for rep in 1:3,mode in 1:3,bi in 1:5;launch!(g.a,g.views[(mode-1)*5+bi],g.dm,g.mw,Val(mode),Val(CONV_BTS[bi]));end
   CUDA.synchronize();elapsed+=(time_ns()-t)*1e-9/3
   got=Float64.(Array(g.out));ok&=reinterpret(UInt64,vec(got))==reinterpret(UInt64,vec(g.gold));err=max(err,maximum(abs.(got.-g.gold)));append!(art,reinterpret(UInt64,vec(got)))
  end
  @assert length(art)==755472;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":743040,\"max_abs_error\":%.17g,\"max_rel_error\":0,\"timing_region\":\"sum24group_15kernelpath_batches_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":755472,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",err,file)
 end
end
