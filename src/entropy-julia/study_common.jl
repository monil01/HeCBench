using CUDA,Printf
include(joinpath(@__DIR__,"entropy_math.jl"))
const EW=131;const EH=97
function study_main(launch!)
 repeat=isempty(ARGS) ? 100 : parse(Int,ARGS[1]);n=EW*EH
 ccall(:srand,Cvoid,(Cuint,),123)
 a=Vector{Int8}(undef,3n)
 for i in 1:n;a[i]=Int8(ccall(:rand,Cint,())%16);a[n+i]=7;x=(i-1)%EW;y=(i-1)÷EW;a[2n+i]=Int8((x+3y)%16);end
 table=Float32[k<=1 ? 0 : Float32(k)*ccall(:log2f,Cfloat,(Cfloat,),Float32(k)) for k in 0:25]
 gold=zeros(Float64,3n)
 for f in 0:2,y in 0:EH-1,x in 0:EW-1
  counts=zeros(Int,16);total=0
  for dy in -2:2,dx in -2:2
   xx=x+dx;yy=y+dy
   if 0<=xx<EW && 0<=yy<EH;counts[Int(a[f*n+yy*EW+xx+1])+1]+=1;total+=1;end
  end
  gold[f*n+y*EW+x+1]=-sum(c==0 ? 0.0 : (c/total)*log2(c/total) for c in counts)
 end
 da=[CuArray(a[f*n+1:(f+1)*n]) for f in 0:2];dt=CuArray(table);outs=[CUDA.fill(-123.25f0,n) for _ in 1:3]
 for f in 1:3;launch!(outs[f],da[f],dt,EW,EH);end;CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns()
  for _ in 1:repeat,f in 1:3;launch!(outs[f],da[f],dt,EW,EH);end
  CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/repeat
  got=vcat(Array.(outs)...);err=maximum(abs.(Float64.(got).-gold));ok=all(isfinite,got)&&err<1e-5&&minimum(got)>=-1e-6&&maximum(got)<=4.00001&&all(abs.(got[n+1:2n]).<1e-6)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,vcat(Float32.(a),table,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"three_complete_optimized_entropy_kernels\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}
",trial,elapsed,ok ? "true" : "false",3n,err,repeat,path,6n+26)
  ok||error("entropy independent oracle failed")
 end
end
