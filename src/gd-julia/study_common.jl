using CUDA
const STUDY_M=257
const STUDY_N=512
const STUDY_ITERS=32
const STUDY_ALPHA=.4f0
const STUDY_LAMBDA=.01f0
function fixture()
 rows=Int32[0];cols=Int32[];values=Float32[];labels=Int32[]
 for i in 0:STUDY_M-1
  y=isodd(i) ? Int32(-1) : Int32(1);push!(labels,y)
  append!(cols,Int32[0,1+mod(37i,511),1+mod(37i+13,511),1+mod(37i+211,511)])
  append!(values,Float32[.5f0*y,-.5,.5,.5]);push!(rows,length(cols))
 end
 rows,cols,values,labels
end
@inline function compute_body!(i,x,grad,rows,cols,values,labels,obj,correct,m)
 if i<m
  xp=0f0
  for j in rows[i+Int32(1)]:rows[i+Int32(2)]-Int32(1)
   @inbounds xp+=values[j+Int32(1)]*x[cols[j+Int32(1)]+Int32(1)]
  end
  @inbounds y=labels[i+Int32(1)]
  v=log(1f0+exp(-xp*y));CUDA.@atomic obj[1]+=v
  prediction=1f0/(1f0+exp(-xp));t=prediction>=.5f0 ? Int32(1) : Int32(-1)
  if y==t;CUDA.@atomic correct[1]+=Int32(1);end
  accum=exp(-y*xp);accum=accum/(1f0+accum)
  for j in rows[i+Int32(1)]:rows[i+Int32(2)]-Int32(1)
   @inbounds col=cols[j+Int32(1)]+Int32(1);temp=-accum*values[j+Int32(1)]*y
   CUDA.@atomic grad[col]+=temp
  end
 end
 nothing
end
function oracle(rows,cols,values,labels)
 x=zeros(Float64,STUDY_N);grad=zeros(Float64,STUDY_N);obj=0.;l2=0.;correct=0;losses=Float64[]
 for _ in 1:STUDY_ITERS
  fill!(grad,0);obj=0.;correct=0;l2=sum(abs2,x)
  for i in 1:STUDY_M
   xp=sum(Float64(values[j])*x[cols[j]+1] for j in rows[i]+1:rows[i+1]);y=labels[i];obj+=log1p(exp(-xp*y));correct+=(xp>=0 ? 1 : -1)==y
   factor=-y/(1+exp(y*xp))
   for j in rows[i]+1:rows[i+1];grad[cols[j]+1]+=factor*values[j];end
  end
  push!(losses,obj/STUDY_M+.5*Float64(STUDY_LAMBDA)*l2)
  x.-=Float64(STUDY_ALPHA).*(grad./STUDY_M.+Float64(STUDY_LAMBDA).*x)
 end
 return vcat(x,grad,[obj,l2,Float64(correct)]),losses
end
function study(run!)
 rows,cols,values,labels=fixture();x=CUDA.zeros(Float32,STUDY_N);grad=CUDA.zeros(Float32,STUDY_N);obj=CUDA.zeros(Float32,1);l2=CUDA.zeros(Float32,1);correct=CUDA.zeros(Int32,1);dr=CuArray(rows);dc=CuArray(cols);dv=CuArray(values);dy=CuArray(labels);template=CUDA.zeros(Float32,256)
 expected,losses=oracle(rows,cols,values,labels);prefix=vcat(Float32.(rows),Float32.(cols),values,Float32.(labels),[STUDY_LAMBDA,STUDY_ALPHA,Float32(STUDY_ITERS)],zeros(Float32,STUDY_N))
 function training!()
  CUDA.fill!(x,0);CUDA.synchronize();t=time_ns()
  for _ in 1:STUDY_ITERS
   CUDA.fill!(obj,0);CUDA.fill!(l2,0);CUDA.fill!(correct,0);CUDA.fill!(grad,0)
   run!(x,grad,dr,dc,dv,dy,obj,correct,l2,Int32(STUDY_M),Int32(STUDY_N),STUDY_LAMBDA,STUDY_ALPHA,template)
  end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9;got=vcat(Array(x),Array(grad),Array(obj),Array(l2),Float32.(Array(correct)));ok=all(isfinite,got)&&all(abs.(Float64.(got).-expected).<=1e-4.+1e-3.*abs.(expected));return got,elapsed,ok,maximum(abs.(Float64.(got).-expected))
 end
 training!()
 for trial in 1:5
  got,elapsed,ok,err=training!();output=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":4113,\"max_abs_error\":",err,",\"timing_region\":\"32_training_iterations_memsets_compute_update_sync\",\"jit_excluded\":true,\"internal_repeats\":32,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":4113,\"byte_order\":\"little\"}}")
  ok||Base.error("SparseLogisticFullStateOracleFailed")
 end
end
