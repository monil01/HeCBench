const CASES=((16,8,8,1,0),(19,9,7,3,0),(16,8,8,3,1),(16,8,8,3,2));const SENT=-8192.25
function study(allocate,launch)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=0
 for(nx,ny,nz,steps,mode)in CASES
  v=nx*ny*nz;raw=[allocate(Float64,v+2) for k in 1:4];a=[view(q,2:v+1) for q in raw];input=copy(fixture[off+1:off+v]);push!(states,(nx,ny,nz,steps,mode,raw,a,input));off+=v
 end
 function run(c)
  nx,ny,nz,steps,mode,raw,a,input=c;v=nx*ny*nz
  for k in 1:4;h=fill(SENT,v+2);k==1&&(h[2:end-1]=input);copyto!(raw[k],h);end
  CUDA.synchronize(CUDA.context());start=time_ns()
  for step in 1:steps,stage in 1:4;launch(stage,a...,Int32(nx),Int32(ny),Int32(nz));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9;arrays=Array.(raw);guards=Float64[];ok=true
  for h in arrays;append!(guards,Float64[h[1],h[end]]);ok&=h[1]===SENT&&h[end]===SENT;end
  vals=vcat([h[2:end-1] for h in arrays]...);masserr=abs(sum(Float64,arrays[1][2:end-1])-sum(Float64,input));ok&=masserr<=1e-10
  return vals,guards,elapsed,ok,masserr
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true;err=0.;masserr=0.
  for c in states
   vals,guards,t,pass,me=run(c);ref=oracle[oi+1:oi+length(vals)];e=maximum(abs.(vals.-ref));ok&=pass&&all(isfinite,vals)&&all(abs.(vals.-ref).<=1e-11 .+1e-11.*abs.(ref));err=max(err,e);masserr=max(masserr,me);oi+=length(vals);append!(out,vals);append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_MASS_DRIFT ",masserr)
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum4_preallocated_periodic_fourlaunch_step_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("CheDiscreteFieldsMassOrExactGuards")
 end
end
