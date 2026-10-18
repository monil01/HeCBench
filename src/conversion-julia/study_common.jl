const TYPES=(CUDA.CUDACore.BFloat16,Float16,Float32,Int32,Int8,UInt8)
const UTYPES=(UInt16,UInt16,UInt32,UInt32,UInt8,UInt8)
@inline cast_value(::Type{T},x::Integer) where {T<:Integer}=x%T
@inline cast_value(::Type{T},x::AbstractFloat) where {T<:Integer}=unsafe_trunc(T,x)
@inline cast_value(::Type{T},x) where {T<:AbstractFloat}=T(x)
function study(allocate,launch)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=0
 while off<length(fixture)
  n,s,t=Int.(fixture[off+1:off+3]);Ts=TYPES[s];Td=TYPES[t];Us=UTYPES[s];Ud=UTYPES[t];bits=Us.(fixture[off+4:off+3+n]);input=collect(reinterpret(Ts,bits));guardS=reinterpret(Ts,Us(0xa5a5a5a5%Us));guardD=reinterpret(Td,Ud(0xa5a5a5a5%Ud));rawin=vcat(guardS,input,guardS);a=allocate(Ts,n+2);b=allocate(Td,n+2);copyto!(a,rawin);push!(states,(n,s,t,a,b,rawin,guardD));off+=3+n
 end
 function run(c)
  n,s,t,a,b,rawin,g=c;fill!(b,g);CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:10;launch(view(b,2:n+1),view(a,2:n+1),Int32(n));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/10;got=Array(b);bits=collect(reinterpret(UTYPES[t],got));ok=reinterpret(UInt8,Array(a))==reinterpret(UInt8,rawin)&&isequal(got[1],g)&&isequal(got[end],g)&&all(isfinite,Float64.(got[2:end-1]));return Float64.(bits[2:end-1]),Float64[bits[1],bits[end]],elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true
  for c in states
   vals,guards,dt,pass=run(c);ok&=pass&&vals==oracle[oi+1:oi+length(vals)];oi+=length(vals);append!(out,vals);append!(out,guards);elapsed+=dt
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum60_typed_conversion_10launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("All30TypedConversionsBitPatternsInputAndGuards")
 end
end
