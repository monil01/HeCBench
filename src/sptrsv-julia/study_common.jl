using Printf
include(joinpath(@__DIR__,"study_cases.jl"))
const G=8
const GUARD=123456789.0
function study(allocate,launch)
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"input_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));cases=CASES;states=[]
 function buf(T,vals)
  host=vcat(fill(T(GUARD),G),T.(vals),fill(T(GUARD),G));raw=allocate(host);(raw=raw,dev=view(raw,G+1:length(raw)-G),host=host)
 end
 for c in cases
  off=Int(c.io);m=Int(c.m);nz=Int(c.nnz);len=Int(c.length);arr=[]
  for (T,n)in [(Int32,m+1),(Int32,nz),(Float64,nz),(Float64,m),(Int32,len)]
   push!(arr,buf(T,f[off+1:off+n]));off+=n
  end
  push!(arr,buf(Float64,zeros(m)));push!(arr,buf(Int32,zeros(Int32,m)));push!(states,(c=c,arr=arr))
 end
 function reset(s)
  copyto!(s.arr[6].raw,s.arr[6].host);copyto!(s.arr[7].raw,s.arr[7].host);CUDA.device_synchronize()
 end
 function capture!(payload,s)
  for a in s.arr;append!(payload,Float64.(Array(a.raw)));end
 end
 function invoke(s)
  a=s.arr;c=s.c;launch(a[1].dev,a[2].dev,a[3].dev,a[7].dev,Int32(c.m),a[4].dev,a[6].dev,a[5].dev,Int32(c.length))
 end
 function checked(s)
  c=s.c;a=s.arr;xx=Array(a[6].raw);ff=Array(a[7].raw);ok=xx[G+1:end-G]==oracle[Int(c.oo)+1:Int(c.oo+c.m)]&&all(==(Int32(1)),ff[G+1:end-G]);ok&=all(==(GUARD),xx[1:G])&&all(==(GUARD),xx[end-G+1:end])&&all(==(Int32(GUARD)),ff[1:G])&&all(==(Int32(GUARD)),ff[end-G+1:end]);ok
 end
 for s in states;invoke(s);CUDA.device_synchronize();checked(s)||error("warmoracle");end
 for trial in 1:5
  pre=Float64[];post=Float64[];seconds=0.;ok=true
  for s in states
   reset(s);capture!(pre,s)
   for r in 1:3
    reset(s);start=time_ns();invoke(s);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9/3;ok&=checked(s);append!(post,Float64.(Array(s.arr[6].raw)));append!(post,Float64.(Array(s.arr[7].raw)))
   end
   capture!(post,s)
   for a in s.arr[1:5];ok&=Array(a.raw)==a.host;end
  end
  append!(pre,post);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"sptrsv_trial$(trial).bin");open(io->write(io,pre),path,"w")
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":3,\"timing_region\":\"sum_five_custom_syncfree_solves_reset_outside_each_interval_completion_per_solve\",\"max_abs_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,ok ? "true" : "false",seconds,path,length(pre),length(pre))
  ok||error("fullsolutionflagsinputguards")
 end
end
