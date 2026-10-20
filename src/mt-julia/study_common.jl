# Own Julia source-aligned MT orchestration; CPU C ABI is an independent checker only.
using Libdl
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'\"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v) for(k,v) in x),",")*"}"
struct Metrics
 checked::UInt64;uniform_mismatches::UInt64;gaussian_failures::UInt64;max_abs_error::Float64;max_bound_fraction::Float64;sum::Float64;sum_squares::Float64;min_value::Float64;max_value::Float64
end
@assert sizeof(Metrics)==72
struct Buffer{T,F,D}
 full::F;data::D;n::Int
end
function allocation(::Type{T},n) where T
 full=CuArray{UInt8,1,CUDA.DeviceMemory}(undef,n*sizeof(T)+512);fill!(full,UInt8(0xa7));CUDA.device_synchronize()
 data=unsafe_wrap(CuArray{T,1,CUDA.DeviceMemory},reinterpret(CUDA.CuPtr{T},pointer(full)+256),n;own=false)
 Buffer{T,typeof(full),typeof(data)}(full,data,n)
end
function guards!(io,b,id)
 lo=Vector{UInt8}(undef,256);hi=similar(lo);copyto!(lo,1,b.full,1,256);copyto!(hi,1,b.full,b.n*4+257,256);CUDA.device_synchronize();write(io,UInt64(id),UInt64(b.n*4),lo,hi)
 all(==(0xa7),lo)&&all(==(0xa7),hi)
end
function read_gpu(r,n)
 a=Vector{Float32}(undef,n);copyto!(a,1,r.data,1,n);CUDA.device_synchronize();a
end
function main()
 @assert length(ARGS)==3
 data,folder,library=ARGS;mkpath(folder);handle=Libdl.dlopen(realpath(library));checker=Libdl.dlsym(handle,:mt_check_abi)
 path=joinpath(data,"MersenneTwister.dat");@assert filesize(path)==65536;original=open(f->read!(f,Vector{UInt32}(undef,16384)),path);@assert filesize(joinpath(data,"MersenneTwister.raw"))==262144
 P=allocation(UInt32,16384);R=allocation(Float32,4096*5860);cases=((5860,UInt32(777),3),(38,UInt32(0),2),(20,UInt32(0xffffffff),2))
 for trial=0:5
  prefix=Float32[];output=Float32[];valid=true;primary=0.;guardio=trial>0 ? open(joinpath(folder,"$(MODEL)_trial$(trial).guards.bin"),"w") : devnull;metricsio=trial>0 ? open(joinpath(folder,"$(MODEL)_trial$(trial).audit.jsonl"),"w") : devnull
  for(c,(n,seed,repeats)) in enumerate(cases)
   parameters=copy(original);parameters[4:4:end].=seed;copyto!(P.data,parameters);fill!(R.data,reinterpret(Float32,UInt32(0xa7a7a7a7)));CUDA.device_synchronize()
   if trial>0
    actual=Array(P.data);valid&=actual==parameters;append!(prefix,Float32[c-1,n,seed&UInt32(65535),seed>>16,repeats,4096,128]);append!(prefix,Float32.(reinterpret(UInt8,actual)))
   else
    actual=UInt32[]
   end
   begin_ns=time_ns()
   for iteration=1:repeats;dispatch_mt!(P.data,R.data,Int32(n));dispatch_boxmuller!(R.data,Int32(n));end
   CUDA.device_synchronize();seconds=(time_ns()-begin_ns)*1e-9/repeats
   if c==1;primary=seconds;end
   if trial==0;continue;end
   count=4096*n;gaussian=read_gpu(R,count);valid&=guards!(guardio,P,(c-1)*2)&&guards!(guardio,R,(c-1)*2+1)
   # Full deterministic replay, explicitly outside original timed intermediate state.
   dispatch_mt!(P.data,R.data,Int32(n));CUDA.device_synchronize();uniform=read_gpu(R,count)
   metrics=Ref(Metrics(0,0,0,0,0,0,0,0,0));status=GC.@preserve actual uniform gaussian begin
    ccall(checker,Cint,(Ptr{UInt32},Ptr{Float32},Ptr{Float32},Cint,Ref{Metrics}),actual,uniform,gaussian,n,metrics)
   end
   valid&=status==0;v=metrics[]
   println(metricsio,json(Dict("case"=>c-1,"nPerRng"=>n,"seed"=>seed,"checked"=>v.checked,"uniform_mismatches"=>v.uniform_mismatches,"gaussian_failures"=>v.gaussian_failures,"max_abs_error"=>v.max_abs_error,"max_bound_fraction"=>v.max_bound_fraction,"pair_mean_seconds"=>seconds,"uniform_capture"=>"untimed_deterministic_MT_replay")))
   append!(output,gaussian);append!(output,uniform)
  end
  if trial==0;continue;end
  close(guardio);close(metricsio);inputs=length(prefix);append!(prefix,output);path="trial$(trial).bin";write(path,prefix)
  println("STUDY_TRIAL "*json(Dict("trial"=>trial,"time_seconds"=>primary,"correct"=>valid,"checked_values"=>length(prefix),"max_abs_error"=>nothing,"max_rel_error"=>nothing,"input_prefix_values"=>inputs,"timing_region"=>"source_full4096x5860_MT_BoxMuller_pair_mean3_only","jit_excluded"=>true,"internal_repeats"=>3,"output_artifact"=>Dict("path"=>path,"dtype"=>"float32","count"=>length(prefix),"byte_order"=>"little"))));flush(stdout);valid||error("fullindependentMT/BoxMulleroracle/guardfailure")
 end
 CUDA.device_synchronize();for b in(P,R);CUDA.unsafe_free!(b.data);CUDA.unsafe_free!(b.full);end
 Libdl.dlclose(handle)
end
