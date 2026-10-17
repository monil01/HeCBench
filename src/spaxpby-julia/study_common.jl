using CUDA,Printf
function fixture(f)
 n=4099;k=(1027,4099,7)[f];idx=sort!(Int64[(37j+11)%n for j in 0:k-1]);x=Float32[(i%17-8)/16 for i in idx];y=Float32[(i%23-11)/8 for i in 0:n-1];a=Float32((1,-.5,2)[f]);b=Float32((1,.5,0)[f]);(idx,x,y,a,b)
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=[fixture(f) for f in 1:3];dy=[CuArray(f[3]) for f in fs];di=[CuArray(f[1]) for f in fs];dx=[CuArray(f[2]) for f in fs];prefix=Float32[];golds=[]
 for f in fs
  idx,x,y,a,b=f;append!(prefix,Float32[length(y),length(x),a,b]);append!(prefix,Float32.(idx));append!(prefix,x);append!(prefix,y)
  # Independent dense Float64 oracle, explicitly rounded after each source stage.
  dense=zeros(Float64,length(y));dense[idx.+1]=Float64.(x);g=copy(y)
  for _ in 1:rep
   g=Float32.(Float64(b).*Float64.(g));g[idx.+1]=Float32.(Float64.(g[idx.+1]).+Float64(a).*dense[idx.+1])
  end
  push!(golds,g)
 end
 for f in 1:3;launch!(dy[f],di[f],dx[f],fs[f][4],fs[f][5]);end;CUDA.synchronize()
 for trial in 1:5
  for f in 1:3;copyto!(dy[f],fs[f][3]);end;CUDA.synchronize();st=time_ns()
  for _ in 1:rep,f in 1:3;launch!(dy[f],di[f],dx[f],fs[f][4],fs[f][5]);end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;got=vcat([Array(d) for d in dy]...);gold=vcat(golds...);err=maximum(abs.(got.-gold));ok=all(isfinite,got)&&err<=1f-6
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"three_sparse_axpby_recurrences_host_dispatch_sync_excludes_reset\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,rep,path,length(art));ok||error("full sparse recurrence oracle")
 end
end
