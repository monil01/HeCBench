const NN=64
@inline function row_body(x,row,ptr,ind,v,work,::Val{W}) where W
 if row<Int32(NN)
  @inbounds first=ptr[row+1];last=ptr[row+2];len=last-first
  if W
   sum=0f0;mn=cld(len,Int32(16))*Int32(16);i=x
   while i<mn
    previous=CUDA.shfl_sync(UInt32(0xffffffff),sum,Int32(16),Int32(16))
    @inbounds sum=i<len ? v[ind[first+i+1]+1] : 0f0
    j=Int32(1)
    while j<Int32(16)
     other=CUDA.shfl_up_sync(UInt32(0xffffffff),sum,j,Int32(16))
     if x>=j;sum+=other;end
     j*=Int32(2)
    end
    sum+=previous;i+=Int32(16)
   end
   total=CUDA.shfl_sync(UInt32(0xffffffff),sum,Int32(16),Int32(16))
   if x==0;@inbounds work[row+1]=total;end
  else
   @inbounds work[row+1]=Float32(len)
  end
 end
 return
end
@inline function intersection_body(x,y,row,ptr,ind,v,work,wi,ws,::Val{W}) where W
 if row<Int32(NN)
  @inbounds j=ptr[row+1]+y;stop=ptr[row+2]
  while j<stop
   @inbounds col=ind[j+1];ni=ptr[row+2]-ptr[row+1];nj=ptr[col+2]-ptr[col+1]
   ref=ni<nj ? row : col;cur=ni<nj ? col : row
   @inbounds ws[j+1]=work[row+1]+work[col+1]
   if x==0
    local_sum=0f0
    @inbounds ip=ptr[ref+1];jp=ptr[cur+1];iend=ptr[ref+2];jend=ptr[cur+2]
    while ip<iend&&jp<jend
     @inbounds a=ind[ip+1];b=ind[jp+1]
     if a==b
      @inbounds local_sum+=W ? v[a+1] : 1f0
      ip+=Int32(1);jp+=Int32(1)
     elseif a<b;ip+=Int32(1)
     else;jp+=Int32(1)
     end
    end
    if local_sum!=0f0;CUDA.@atomic wi[j+1]+=local_sum;end
   end
   j+=Int32(4)
  end
 end
 return
end
@inline function finish_body(j,vals,wi,ws,wj)
 if j<=length(vals);@inbounds wj[j]=(0.46f0*vals[j])*(wi[j]/(ws[j]-wi[j]));end
 return
end
function study(run!)
 ptr=Int32[0];ind=Int32[];vals=Float32[]
 for row in 0:NN-1
  for col in 0:NN-1
   if mod(col-row,NN)<17+mod(row,15);push!(ind,Int32(col));push!(vals,Float32(1+mod(3*row+col,9)));end
  end
  push!(ptr,Int32(length(ind)))
 end
 e=length(ind);P=CuArray(ptr);I=CuArray(ind);V=CuArray(vals);W=CUDA.zeros(Float32,NN);WI=CUDA.zeros(Float32,e);WS=similar(WI);WJ=similar(WI)
 wants=Vector{Float32}[]
 for weighted in (true,false)
  vertex=weighted ? Float64[(i+1)/e for i in 0:NN-1] : ones(Float64,NN);sets=[Set(ind[Int(ptr[row+1])+1:Int(ptr[row+2])]) for row in 0:NN-1];rw=[sum(vertex[Int(c)+1] for c in x) for x in sets];ii=Float64[];ss=Float64[];jj=Float64[]
  for row in 0:NN-1,j in Int(ptr[row+1])+1:Int(ptr[row+2])
   col=Int(ind[j]);inter=sum(vertex[Int(c)+1] for c in intersect(sets[row+1],sets[col+1]));s=rw[row+1]+rw[col+1];push!(ii,inter);push!(ss,s);push!(jj,Float64(0.46f0)*Float64(vals[j])*inter/(s-inter))
  end
  push!(wants,Float32.(vcat(rw,ii,ss,jj)))
 end
 for weighted in (true,false);run!(P,I,V,W,WI,WS,WJ,Val(weighted));end;CUDA.synchronize()
 for trial in 1:5
  total=0.;got=Float32[];want=Float32[]
  for (case,weighted) in enumerate((true,false))
   CUDA.synchronize();t=time_ns();for k in 1:10;run!(P,I,V,W,WI,WS,WJ,Val(weighted));end;CUDA.synchronize();total+=(time_ns()-t)*1e-9/10;append!(got,vcat(Array(W),Array(WI),Array(WS),Array(WJ)));append!(want,wants[case])
  end
  err=maximum(abs.(got.-want));ok=all(isfinite,got)&&all(abs.(got.-want).<=1f-5 .+eps(Float32).*abs.(want));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_weighted_unweighted_fivekernel_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Jaccard setintersection oracle failed")
 end
end
