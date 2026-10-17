const STUDY_N=257
const STUDY_ITERATIONS=64
function make_graph(irregular)
 n=STUDY_N;pages=zeros(Int32,n*n);degrees=zeros(UInt32,n);edges=Vector{Vector{Int}}(undef,n)
 for i in 0:n-1
  targets=irregular ? Set([mod(i+1,n),mod(i+13,n),mod(17i+5,n)]) : Set([mod(i+1,n),mod(i+13,n),mod(i+37,n)])
  if irregular
   i%3==0&&push!(targets,mod(i+37,n));i!=0&&push!(targets,0);delete!(targets,i)
  end
  edges[i+1]=sort!(collect(targets));degrees[i+1]=length(targets)
  for j in targets;pages[i*n+j+1]=1;end
 end
 return pages,degrees,edges
end
function independent_expected(pages,degrees,edges)
 n=STUDY_N;rank=fill(Float64(1f0/Float32(n)),n);history=zeros(Float64,n);trajectory=Float64[];old=copy(rank);damping=Float64(.85f0);teleport=Float64((1f0-.85f0)/Float32(n))
 for iteration in 1:STUDY_ITERATIONS
  old=rank;incoming=zeros(Float64,n)
  for i in 1:n,j in edges[i];incoming[j+1]+=old[i]/Float64(degrees[i]);end
  rank=teleport.+damping.*incoming;history=max.(history,abs.(rank.-old));append!(trajectory,rank)
 end
 map=Float64[pages[i*n+j+1]*(old[i+1]/Float64(degrees[i+1])) for i in 0:n-1 for j in 0:n-1]
 return Float32.(vcat(map,trajectory,history))
end
function study(run!)
 cases=[]
 for irregular in (false,true)
  pages,degrees,edges=make_graph(irregular);expected=independent_expected(pages,degrees,edges);push!(cases,(CuArray(pages),CuArray(degrees),CUDA.zeros(Float32,STUDY_N),CUDA.zeros(Float32,STUDY_N*STUDY_N),CUDA.zeros(Float32,STUDY_N),expected))
 end
 function case!(args)
  pages,degrees,rank,map,history,expected=args;CUDA.fill!(rank,1f0/Float32(STUDY_N));CUDA.fill!(history,0);CUDA.synchronize();trajectory=Float32[];total=0.
  for iteration in 1:STUDY_ITERATIONS
   CUDA.synchronize();t=time_ns();run!(pages,rank,map,degrees,Int32(STUDY_N),history);CUDA.synchronize();total+=(time_ns()-t)*1e-9
   observed_history=Array(history);all(isfinite,observed_history)||Base.error("Nonfinitehistoricaldiagnostic");append!(trajectory,Array(rank))
  end
  got=vcat(Array(map),trajectory,Array(history));error=maximum(abs.(Float64.(got).-Float64.(expected)));ok=all(isfinite,got)&&error<1e-6
  return got,ok,error,total
 end
 for args in cases;case!(args);end
 for trial in 1:5
  got=Float32[];ok=true;error=0.;total=0.
  for args in cases;output,good,err,elapsed=case!(args);append!(got,output);ok&=good;error=max(error,err);total+=elapsed;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":165508,\"max_abs_error\":",error,",\"timing_region\":\"sum_two_graphs_64_periteration_map_reduce_sync_totals\",\"jit_excluded\":true,\"internal_repeats\":64,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":165508,\"byte_order\":\"little\"}}")
  ok||Base.error("FullPageRanktrajectory/maps/historyoraclefailed")
 end
end
