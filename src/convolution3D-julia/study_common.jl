const NC=2;const CC=3;const MC=5;const WIN=37;const HIN=35;const KS=3;const HOUT=HIN-KS+1;const WOUT=WIN-KS+1;const WG=cld(WOUT,16);const ZZ=WG*cld(HOUT,16)
const DIMS=(Int32(CC),Int32(MC),Int32(KS),Int32(HIN),Int32(WIN),Int32(HOUT),Int32(WOUT),Int32(WG))
@inline function conv_body(lx,ly,bx,by,bz,x,w,y,C,M,K,Hin,Win,Hout,Wout,Wgrid,::Val{STYLE}) where STYLE
 n=STYLE==1 ? bx : STYLE==2 ? bz : by;m=STYLE==1 ? by : STYLE==2 ? bx : bz;tile=STYLE==1 ? bz : STYLE==2 ? by : bx
 h=(tile÷Wgrid)*Int32(16)+ly;col=mod(tile,Wgrid)*Int32(16)+lx
 if h<Hout&&col<Wout
  s=0f0
  for c in Int32(0):C-Int32(1),p in Int32(0):K-Int32(1),q in Int32(0):K-Int32(1)
   @inbounds s+=x[((n*C+c)*Hin+h+p)*Win+col+q+1]*w[((m*C+c)*K+p)*K+q+1]
  end
  @inbounds y[((n*M+m)*Hout+h)*Wout+col+1]=s
 end
 return
end
grid(style)=style==1 ? (NC,MC,ZZ) : style==2 ? (MC,ZZ,NC) : (ZZ,NC,MC)
function study(run!)
 x=Float32[mod(i,17)-8 for i in 0:NC*CC*HIN*WIN-1];w=Float32[mod(i,13)-6 for i in 0:MC*CC*KS*KS-1];want=Float32[]
 for n in 0:NC-1,m in 0:MC-1,h in 0:HOUT-1,col in 0:WOUT-1
  total=Int64(0)
  for c in 0:CC-1,p in 0:KS-1,q in 0:KS-1
   xi=((n*CC+c)*HIN+h+p)*WIN+col+q;wi=((m*CC+c)*KS+p)*KS+q;total+=(mod(xi,17)-8)*(mod(wi,13)-6)
  end
  push!(want,Float32(total))
 end
 X=CuArray(x);W=CuArray(w);Y=CUDA.fill(-1f0,length(want))
 for style in 1:3;run!(X,W,Y,DIMS,Val(style));end;CUDA.synchronize()
 for trial in 1:5
  total=0.;got=Float32[];wantall=Float32[]
  for style in 1:3
   fill!(Y,-1f0);CUDA.synchronize();start=time_ns();for repeat in 1:10;run!(X,W,Y,DIMS,Val(style));end;CUDA.synchronize();total+=(time_ns()-start)*1e-9/10;append!(got,Array(Y));append!(wantall,want)
  end
  err=maximum(abs.(got.-wantall));ok=all(isfinite,got)&&err<=1f-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":34650,\"max_abs_error\":",err,",\"timing_region\":\"sum_three_native_grid_order_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":34650,\"byte_order\":\"little\"}}")
  ok||error("Convolution integeroracle failed")
 end
end
