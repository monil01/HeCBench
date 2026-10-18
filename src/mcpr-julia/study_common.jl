function load_fixture(name)
 text=read(joinpath(@__DIR__,"study_fixture.h"),String);pattern=Regex(name*"\\[\\d+\\]=\\{([^}]*)\\}");m=match(pattern,text);m===nothing&&error("MissingFixtureArray",name);return parse.(Float64,split(m.captures[1],','))
end
function mathematical_oracle(A,R,n,M)
 counts=zeros(Int64,n*21)
 for row in 0:n-1,sample in 0:M-1
  values=[A[row*21+k+1]+R[sample*21+k+1] for k in 0:20];maximumvalue=maximum(values);best=values[21]==maximumvalue ? 21 : findfirst(==(maximumvalue),values);counts[row*21+best]+=1
 end
 return Float64.(counts)./M,counts
end
transpose_layout(x,rows)=vec(permutedims(reshape(x,21,rows)))
canonical_output(x,n)=vec(permutedims(reshape(x,n,21)))
function study(launch,allocate)
 cases=[];prefix=Float64[]
 for (id,n,M) in ((1,193,97),(2,5,3))
  A=load_fixture("ALPHAS$id");R=load_fixture("DRAWS$id");AS=transpose_layout(A,n);RS=transpose_layout(R,M);want,counts=mathematical_oracle(A,R,n,M);dd=allocate(Float64,length(A));dr=allocate(Float64,length(R));ds=allocate(Float64,length(AS));dt=allocate(Float64,length(RS));copyto!(dd,A);copyto!(dr,R);copyto!(ds,AS);copyto!(dt,RS);probs=[allocate(Float64,length(A)) for _ in 1:3]
  for x in (A,R,AS,RS);append!(prefix,x);end
  append!(prefix,zeros(Float64,3length(A)));append!(prefix,[n,21,M,3,192,96,32256]);push!(cases,(id,n,M,dd,dr,ds,dt,probs,want,counts))
 end
 function case!(c)
  id,n,M,dd,dr,ds,dt,probs,want,counts=c;got=Float64[];times=Float64[];ok=true;err=0.
  for variant in 1:3
   P=probs[variant];CUDA.fill!(P,0.);A=variant==1 ? dd : ds;R=variant==1 ? dr : dt;CUDA.synchronize(CUDA.context());start=time_ns()
   for _ in 1:3;launch(variant,A,R,P,Int32(n),Int32(M));end
   CUDA.synchronize(CUDA.context());push!(times,(time_ns()-start)*1e-9/3);out=Array(P);if variant!=1;out=canonical_output(out,n);end
   err=max(err,maximum(abs.(out.-want)));ok&=maximum(abs.(out.-want))<=.001&&round.(Int64,out.*M)==counts
   for row in 0:n-1;ok&=abs(sum(@view out[row*21+1:row*21+21])-1)<=1e-12;end
   append!(got,out)
  end
  return got,times,ok,err
 end
 for c in cases;got,times,ok,err=case!(c);ok||error("WarmupMCPRMismatch");end
 for trial in 1:5
  out=copy(prefix);components=Float64[];ok=true;err=0.
  for c in cases;got,times,pass,e=case!(c);append!(out,got);append!(components,times);ok&=pass;err=max(err,e);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(components),",\"component_times_seconds\":[",join(components,','),"],\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum_six_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullThreeLayoutsCountsSimplexMismatch")
 end
end
