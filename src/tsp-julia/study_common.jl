const TSP_G=64
const TSP_GUARD=Int32(reinterpret(Int32,UInt32(0xdeadbeef)))
function readwords(name)
 p=joinpath(@__DIR__,name);filesize(p)%4==0||error("wordfile size")
 open(io->read!(io,Vector{UInt32}(undef,filesize(p)÷4)),p)
end
function study(allocate,launch,diagnostic)
 data=readwords("fixtures_u32.bin");oracle=readwords("oracle_u32.bin");idx=2;cases=Any[];Int(data[1])==2||error("case count")
 for _ in 1:2
  n=Int(data[idx]);r=Int(data[idx+1]);t=Int(data[idx+2]);idx+=3;xs=data[idx:idx+n-1];idx+=n;ys=data[idx:idx+n-1];idx+=n;stride=((3n+2+31)÷32)*32
  all=map((n,n,stride*r,1,1,n*n)) do count
   d=allocate(count+2TSP_G);copyto!(d,fill(TSP_GUARD,count+2TSP_G));d
  end
  di,dj,dg,dc,db,dp=all;x=reinterpret(Float32,view(di,TSP_G+1:TSP_G+n));y=reinterpret(Float32,view(dj,TSP_G+1:TSP_G+n));copyto!(x,collect(reinterpret(Float32,xs)));copyto!(y,collect(reinterpret(Float32,ys)))
  push!(cases,(n=n,r=r,t=t,stride=stride,x=x,y=y,g=view(dg,TSP_G+1:TSP_G+stride*r),climbs=view(dc,TSP_G+1:TSP_G+1),best=view(db,TSP_G+1:TSP_G+1),pairs=view(dp,TSP_G+1:TSP_G+n*n),all=all))
 end
 idx==length(data)+1||error("fixture trailing data")
 function reset(c)
  fill!(c.g,TSP_GUARD);fill!(c.climbs,Int32(0));fill!(c.best,typemax(Int32));fill!(c.pairs,TSP_GUARD);CUDA.device_synchronize(blocking=true)
 end
 for c in cases;reset(c);launch(c);CUDA.device_synchronize(blocking=true);diagnostic(c);CUDA.device_synchronize(blocking=true);end
 for trial in 1:5
  prefix=UInt32[2];suffix=UInt32[];times=Float64[]
  for c in cases
   reset(c);start=time_ns();launch(c);CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
   diagnostic(c);CUDA.device_synchronize(blocking=true)
   append!(prefix,UInt32[c.n,c.r,c.t]);append!(prefix,reinterpret(UInt32,Array(c.x)));append!(prefix,reinterpret(UInt32,Array(c.y)))
   append!(suffix,reinterpret(UInt32,Array(c.g)));append!(suffix,reinterpret(UInt32,Array(c.climbs)));append!(suffix,reinterpret(UInt32,Array(c.best)));append!(suffix,reinterpret(UInt32,Array(c.pairs)))
   for d in c.all
    h=Array(d);append!(suffix,reinterpret(UInt32,view(h,1:TSP_G)));append!(suffix,reinterpret(UInt32,view(h,length(h)-TSP_G+1:length(h))))
   end
  end
  ok=suffix==oracle&&prefix==data;out=vcat(prefix,suffix);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum2_source_tiled_twoopt_kernels_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":2,\"untimed_diagnostic_kernel_calls\":2,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("TSP fullscratch/tours/edges/climbs/best/distance/input/guard oracle mismatch")
 end
end
