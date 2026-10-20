const ROOT=@__DIR__
const CHECKLIB=joinpath(ROOT,"study_check.so")
function alloc(::Type{T},n)where T
 storage=CuArray{T}(undef,n+64);(storage=storage,data=view(storage,33:n+32),n=n)
end
function reset!(s)
 for b in(s.result,s.counts,s.echo);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(b.storage)),UInt8(165),length(b.storage)*sizeof(eltype(b.storage)));end
 CUDA.device_synchronize();nothing
end
function run_main(s,ch)
 CUDA.device_synchronize();start=time_ns();launch_main(s,ch);CUDA.device_synchronize();(time_ns()-start)*1e-9
end
function capture(s,folder,trial,ch)
 result=Array(s.result.storage);counts=Array(s.counts.storage);echo=Array(s.echo.storage);missing=Ref{Cint}(0);extra=Ref{Cint}(0)
 ok=GC.@preserve result counts ccall((:interval_check_coverage,CHECKLIB),Cint,(Ptr{Cvoid},Ptr{Int32},Ref{Cint},Ref{Cint}),pointer(result,33),pointer(counts,33),missing,extra)==1
 println(stderr,"trial",trial," choice",ch," exactrootcoverage_missing",missing[]," extra_candidates",extra[])
 guards=UInt8[]
 for a in(result,counts,echo);v=reinterpret(UInt8,a);n=32*sizeof(eltype(a));append!(guards,view(v,1:n));append!(guards,view(v,length(v)-n+1:length(v)));end
 ok&=all(==(UInt8(165)),guards);values=Float64[]
 for i in 0:65535
  k=Int(counts[33+i]);push!(values,k);if !(0<=k<=128);ok=false;continue;end
  intervals=[(result[33+j*65536+i].lo,result[33+j*65536+i].hi) for j in 0:k-1]
  sort!(intervals)
  for (lo,up)in intervals;push!(values,lo);push!(values,up);ok&=isfinite(lo)&&isfinite(up)&&lo<=up;end
 end
 path=joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_choice$(ch)")
 for(suffix,a)in((".intervals.bin",result),(".counts.i32",counts),(".echo.f64",echo),(".guards.u8",guards));open(io->write(io,a),path*suffix,"w");end
 println("STUDY_RAW_ARTIFACT {\"trial\":",trial,",\"choice\":",ch,",\"path\":\"",path,"\",\"result_bytes\":",sizeof(IV)*length(result),",\"guard_bytes\":",length(guards),"}")
 values,ok
end
function main()
 length(ARGS)==1||error("persistent raw folder required");folder=ARGS[1];mkpath(folder);sizeof(IV)==16||error("interval ABI");s=(result=alloc(IV,65536*128),counts=alloc(Int32,65536),echo=alloc(Float64,2*65536+4),domain=IV(Float64(.01f0),4.))
 for ch in 0:1;reset!(s);run_main(s,ch);launch_echo(s);CUDA.device_synchronize();end
 for trial in 1:5
  prefix=Float64[];values=Float64[];seconds=0.;ok=true
  for ch in 0:1
   reset!(s);launch_echo(s);CUDA.device_synchronize();e=Array(s.echo.data);append!(prefix,e);append!(prefix,Float64[ch,65536,1,1024,64]);seconds+=run_main(s,ch);out,pass=capture(s,folder,trial,ch);append!(values,out);ok&=pass
  end
  append!(prefix,values);path="trial$(trial).bin";open(io->write(io,prefix),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(prefix),",\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum2_full65536equation_Newton_variants_kernel_and_contextcompletion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(prefix),",\"byte_order\":\"little\"}}")
  ok||error("independent rootcoverage/shape/guard failed")
 end
end
