# All4 computational stages, no GDAL I/O. CPU source oracle plus independent controls.
const STSG_GUARD=32
function stsg_cases()
 f=reinterpret(Float64,read(joinpath(@__DIR__,"study_fixture.bin")));off=2;cases=[]
 for ci in 1:Int(f[1])
  head=copy(f[off:off+7]);X,Y,B,Z,w=Int32.(head[1:5]);off+=8;N=X*Y;W=(2w+1)^2;Q=N*B*Z
  raw=Int16.(f[off:off+Q-1]);off+=Q;qa=UInt8.(f[off:off+Q-1]);off+=Q
  # Payload guard arrays allocated independently; allinitial values are device-read back.
  inputs=Any[raw,qa,zeros(Float32,Q),zeros(Float32,Q),zeros(Float32,N*B),zeros(Float32,N*W*4),zeros(Float32,N*W*3),zeros(Float32,Q),zeros(Float32,N*B),zeros(Int32,N*B),zeros(Int32,N*W*3)]
  guarded=map(inputs) do v
   sentinel=eltype(v)(eltype(v)==UInt8 ? 173 : -37)
   vcat(fill(sentinel,STSG_GUARD),v,fill(sentinel,STSG_GUARD))
  end
  push!(cases,(head,guarded))
 end
 @assert off==length(f)+1
 return cases
end
function stsg_study(factory)
 cases=stsg_cases();oracle=reinterpret(Float64,read(joinpath(@__DIR__,"study_oracle.bin")))
 for trial in 0:5
  prefix=Float64[length(cases)];got=Float64[];extra=Float64[];seconds=0.;ok=true;independent_error=0.
  for (head,inputs) in cases
   X,Y,B,Z,w=Int32.(head[1:5]);cosy=Float32(head[6]);samp=Float32(head[7]);snow=Int32(head[8]);Q=X*Y*B*Z
   # Observer executes the same complete pipeline with intermediate copies outside ROI.
   # Timed invocation has fresh identical allocations/initialization and no intermediate D2H.
   snapshots=Vector{Vector{Float64}}()
   for observer in (true,false)
    base=CuArray.(inputs);v=map(a->view(a,STSG_GUARD+1:length(a)-STSG_GUARD),base)
    raw,rawqa,nd,qa,ref,res,r3,out,vin,res1,index=v
    if !observer
     append!(prefix,head)
     for (b,h)in zip(base,inputs)
      actual=Array(b);ok&=actual==h;append!(prefix,actual)
     end
    end
    launch! = factory(X,Y)
    CUDA.synchronize(CUDA.context());start=time_ns()
    launch!(1,raw,rawqa,X,Y,B,Z,nd,qa)
    if observer
     CUDA.synchronize(CUDA.context());a=Array(nd);q=Array(qa);push!(snapshots,vcat(Float64.(a),Float64.(q)))
     # Independent conversion/QA contract over every actual converted element.
     hraw=Array(raw);hq=Array(rawqa)
     for i in eachindex(a)
      want=Float32(hraw[i])/10000f0;wantq=(want<-.2f0||want>1f0||hq[i]>3) ? -1f0 : Float32(hq[i])
      independent_error=max(independent_error,abs(Float64(a[i])-Float64(want)));ok&=abs(Float64(a[i])-Float64(want))<=2e-7 && q[i]==wantq
     end
    end
    launch!(2,cosy,Int32(2),nd,qa,X,Y,B,Z,ref,r3,res1)
    if observer
     CUDA.synchronize(CUDA.context());push!(snapshots,reduce(vcat,(Float64.(Array(b)) for b in (nd,qa,ref,r3,res1))))
    end
    fill!(r3,0f0)
    launch!(3,nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,w,res)
    if observer
     CUDA.synchronize(CUDA.context());push!(snapshots,Float64.(Array(res)))
    end
    launch!(4,nd,qa,ref,Int32(0),Y,Int32(0),Int32(0),X,Y,B,Z,w,samp,snow,out,vin,res,r3,index)
    CUDA.synchronize(CUDA.context())
    if !observer
     seconds+=(time_ns()-start)*1e-9
     for a in snapshots;append!(got,a);end
     for b in (nd,qa,ref,res,r3,out,vin,res1,index);append!(got,Array(b));end
     # Manufactured low-NDVI threshold control is independent of source CPU implementation.
     if X==7;ok&=all(iszero,Array(out));end
     for (b,h)in zip(base,inputs)
      a=Array(b);ok&=a[1:STSG_GUARD]==h[1:STSG_GUARD] && a[end-STSG_GUARD+1:end]==h[end-STSG_GUARD+1:end]
      append!(extra,a[1:STSG_GUARD]);append!(extra,a[end-STSG_GUARD+1:end])
     end
     ok&=Array(base[1])==inputs[1] && Array(base[2])==inputs[2]
     append!(extra,Array(base[1]));append!(extra,Array(base[2]))
    end
    for b in base;CUDA.unsafe_free!(b);end
   end
  end
  @assert length(got)==length(oracle)
  err=0.;rel=0.
  for i in eachindex(got)
   e=abs(got[i]-oracle[i]);err=max(err,e);rel=max(rel,e/max(abs(oracle[i]),1e-30));ok&=isfinite(got[i])&&e<=2e-5+2e-5*abs(oracle[i])
  end
  if trial==0;ok||error("WarmupOracleMismatch maxabs=$err independent=$independent_error");continue;end
  data=vcat(prefix,got,extra);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,data)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"independent_conversion_max_abs_error\":%.17g,\"timing_region\":\"sum3cases_full4stage_pipeline_source_scratch_memset_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(data),err,rel,independent_error,path,length(data))
  ok||error("FullOracleMismatch")
 end
end
