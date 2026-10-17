const ITERATIONS=19
const GAIN=0.1f0
const THRESHOLD=0.00001f0
const WIDTHS=(65,257)
function fixture(w)
 psf=zeros(Float32,49)
 for y in -1:1,x in -1:1
  psf[(y+3)*7+x+4]=(x==0&&y==0) ? 1f0 : (y==0 ? 0.125f0 : (x==0 ? -0.0625f0 : 0.03125f0))
 end
 dirty=zeros(Float32,w*w)
 sources=w==65 ? ((0,0,4f0),(32,20,-3f0),(64,50,2f0)) : ((32769%w,32769÷w,4f0),(300%w,300÷w,4f0))
 for (sx,sy,a) in sources,y in 0:w-1,x in 0:w-1
  px=x-sx+3;py=y-sy+3
  if 0<=px<7&&0<=py<7;dirty[y*w+x+1]+=a*psf[py*7+px+1];end
 end
 return dirty,psf
end
priority(i)=(i%32768,i÷32768)
function mathematical_oracle(dirty,psf,w)
 residual=copy(dirty);model=zeros(Float32,length(dirty));trace=Float64[]
 for _ in 1:ITERATIONS
  best=0;val=0f0
  for i in 0:length(residual)-1
   v=residual[i+1]
   if abs(v)>abs(val)||(abs(v)==abs(val)&&priority(i)<priority(best));best=i;val=v;end
  end
  @assert abs(val)>THRESHOLD
  push!(trace,best,val);factor=GAIN*val;model[best+1]+=factor;sx=best%w;sy=best÷w
  for y in max(0,sy-3):min(w-1,sy+3),x in max(0,sx-3):min(w-1,sx+3)
   residual[y*w+x+1]=fma(-factor,psf[(y-sy+3)*7+x-sx+4],residual[y*w+x+1])
  end
 end
 return vcat(Float64.(model),Float64.(residual),Float64[24,1],trace)
end
function study(peak_launch,subtract_launch,allocate)
 cases=[];prefix=Float64[]
 for (caseid,w) in enumerate(WIDTHS)
  dirty,psf=fixture(w);append!(prefix,dirty);append!(prefix,psf);append!(prefix,zeros(Float32,length(dirty)));append!(prefix,[w,7,ITERATIONS,GAIN,THRESHOLD,caseid]);want=mathematical_oracle(dirty,psf,w)
  dr=allocate(Float32,length(dirty));dp=allocate(Float32,49);pk=allocate(Peak,128);copyto!(dp,psf);push!(cases,(w,dirty,psf,want,dr,dp,pk,zeros(Float32,length(dirty)),zeros(Float64,2ITERATIONS)))
 end
 function find_peak!(img,peaks)
  peak_launch(img,UInt64(length(img)),peaks);vals=Array(peaks);best=Peak(0,0f0)
  for v in vals;if abs(v.val)>abs(best.val);best=v;end;end
  return best
 end
 function case!(c)
  w,dirty,psf,want,res,dp,pk,model,trace=c;copyto!(res,dirty);fill!(model,0f0);psfpeak=find_peak!(dp,pk);CUDA.synchronize(CUDA.context());t=time_ns()
  for it in 1:ITERATIONS
   peak=find_peak!(res,pk);pos=Int(peak.pos);trace[2it-1]=pos;trace[2it]=peak.val
   rx=pos%w;ry=pos÷w;px=Int(psfpeak.pos)%7;py=Int(psfpeak.pos)÷7
   sx=max(0,rx-px);sy=max(0,ry-py);ex=min(w-1,rx+6-px);ey=min(w-1,ry+6-py)
   subtract_launch(dp,res,Int32(w),Int32(sx),Int32(sy),Int32(ex),Int32(ey),Int32(rx-px),Int32(ry-px),peak.val,GAIN)
   model[pos+1]+=peak.val*GAIN
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9
  got=vcat(Float64.(model),Float64.(Array(res)),Float64[psfpeak.pos,psfpeak.val],trace);err=maximum(abs.(got.-want));ok=err<=0.001&&all(abs(trace[i])>THRESHOLD for i in 2:2:length(trace))&&all(got[2w*w+i]==want[2w*w+i] for i in 3:2:2+2ITERATIONS)
  return got,elapsed,ok,err
 end
 for c in cases;case!(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;err=0.
  for c in cases;got,t,pass,e=case!(c);append!(out,got);elapsed+=t;ok&=pass;err=max(err,e);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum_two_19_iteration_clean_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullCleanModelResidualPeakHistoryMismatch")
 end
end
