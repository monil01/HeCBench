using Libdl
const LIBBLAS_STUDY="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
const MM=32;const NN=32;const KK=16
@inline bfencode(x::Float32)=UInt16((reinterpret(UInt32,x)+UInt32(0x7fff)+((reinterpret(UInt32,x)>>16)&UInt32(1)))>>16)
@inline bfdecode(x::UInt16)=reinterpret(Float32,UInt32(x)<<16)
function callgemm(handle,A,B,C,atype,ctype,compute,alpha,beta)
 GC.@preserve A B C alpha beta begin
  return ccall((:cublasGemmEx,LIBBLAS_STUDY),Cint,(Ptr{Cvoid},Cint,Cint,Cint,Cint,Cint,Ptr{Cvoid},CuPtr{Cvoid},Cint,Cint,CuPtr{Cvoid},Cint,Cint,Ptr{Cvoid},CuPtr{Cvoid},Cint,Cint,Cint,Cint),handle,0,0,NN,MM,KK,alpha,pointer(B),atype,NN,pointer(A),atype,KK,beta,pointer(C),ctype,NN,compute,-1)
 end
end
function study(allocate)
 handle=Ref{Ptr{Cvoid}}(C_NULL);status=ccall((:cublasCreate_v2,LIBBLAS_STUDY),Cint,(Ref{Ptr{Cvoid}},),handle);status==0||error("cublasCreate failed",status);version=Ref{Cint}();ccall((:cublasGetVersion_v2,LIBBLAS_STUDY),Cint,(Ptr{Cvoid},Ref{Cint}),handle[],version);println("STUDY_LIBRARY path=",realpath(LIBBLAS_STUDY)," version=",version[])
 specs=((Float64,1,1,70,:f64),(Float32,0,0,68,:f32_tf32_label),(Float32,0,0,68,:f32_bf16_label),(Float32,0,0,68,:f32_f16_label),(Float32,0,0,68,:f32),(UInt16,2,2,64,:f16_acc16),(UInt16,2,2,68,:f16_acc32),(UInt16,14,14,68,:bf16),(Int8,3,10,72,:int8))
 cases=[];flags=Int[];expectedall=Float64[]
 for (T,atype,ctype,compute,label) in specs
  aa=Float32[Float32(mod(i,17)-8)/32f0 for i in 0:MM*KK-1];bb=Float32[Float32(mod(i,13)-6)/32f0 for i in 0:KK*NN-1]
  if T==UInt16
   a=atype==2 ? collect(reinterpret(UInt16,Float16.(aa))) : bfencode.(aa);b=atype==2 ? collect(reinterpret(UInt16,Float16.(bb))) : bfencode.(bb)
  elseif T==Int8
   a=Int8[Int8(mod(i,17)-8) for i in 0:MM*KK-1];b=Int8[Int8(mod(i,13)-6) for i in 0:KK*NN-1]
  else
   a=T.(aa);b=T.(bb)
  end
  S=T==Int8 ? Int32 : T;A=allocate(T,length(a));B=allocate(T,length(b));C=allocate(S,MM*NN);copyto!(A,a);copyto!(B,b);fill!(C,zero(S));CUDA.device_synchronize(blocking=true)
  scalar=compute==70 ? Float64 : compute==64 ? UInt16 : compute==72 ? Int32 : Float32;alpha=Ref{scalar}(compute==64 ? UInt16(0x3c00) : one(scalar));beta=Ref{scalar}(zero(scalar));args=(handle[],A,B,C,atype,ctype,compute,alpha,beta);s=callgemm(args...);CUDA.device_synchronize(blocking=true);push!(flags,Int(s));println("STUDY_CASE label=",label," status=",s," compute=",compute)
  (s==0||(label==:bf16&&s==15))||error("UnexpectedcuBLASstatus",s)
  if s==0
   want=Float64[]
   for row in 0:MM-1,col in 0:NN-1
    z=sum((mod(row*KK+k,17)-8)*(mod(k*NN+col,13)-6) for k in 0:KK-1);value=label==:int8 ? Float64(z) : Float64(z)/1024;push!(want,value)
   end
   append!(expectedall,want);push!(cases,(args,label))
  end
 end
 for trial in 1:5
  total=0.;got=Float64.(flags)
  for (args,label) in cases
   for k in 1:10
    CUDA.device_synchronize(blocking=true);t=time_ns();s=callgemm(args...);CUDA.device_synchronize(blocking=true);total+=(time_ns()-t)*1e-9/10;s==0||error("TimedcuBLASstatus",s)
   end
   data=Array(args[4]);values=label in (:f16_acc16,:f16_acc32) ? Float64.(reinterpret(Float16,data)) : label==:bf16 ? Float64.(bfdecode.(data)) : Float64.(data);append!(got,values)
  end
  expected=vcat(Float64.(flags),expectedall);err=maximum(abs.(got.-expected));ok=all(isfinite,got)&&err<=.004;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_supported_gemmex_vendor_call_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Integer-dot-derived GEMMCPUoracle failed")
 end
 ccall((:cublasDestroy_v2,LIBBLAS_STUDY),Cint,(Ptr{Cvoid},),handle[])
end
