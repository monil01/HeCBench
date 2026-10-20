# cuMF_SGD source operators, translated to Julia by Codex2026-10-19.
# Original source/notices preserved in original_native/. No foreign kernel call.
@inline lcg_next(x::UInt32)=(UInt32(26757677)*x+UInt32(1))&UInt32(0x7fffffff)
@inline function source_random!(state,index)
 @inbounds value=lcg_next(state[index]);@inbounds state[index]=value
 Float32(value)/2147483648.0f0
end
@inline function body_rng_sized!(gid,seed,state,n)
 if gid<n;@inbounds state[gid+Int32(1)]=lcg_next(seed⊻UInt32(gid));end
 nothing
end
@inline function body_rng_unsized!(gid,seed,state)
 @inbounds state[gid+Int32(1)]=lcg_next(seed⊻UInt32(gid));nothing
end
@inline function body_features!(gid,stride,state,state_size,features,array_size,k,scale)
 index=gid;state_id=rem(gid,state_size)+Int32(1)
 @inbounds while Int64(index)<array_size
  features[index+Int32(1)]=Float16(source_random!(state,state_id)*scale)
  index+=stride
 end;nothing
end
@inline function body_transform!(gid,stride,input,output,n)
 index=Int64(gid)
 @inbounds while index<n
  output[index+Int64(1)]=Float32(input[index+Int64(1)]);index+=Int64(stride)
 end;nothing
end
@inline function source_update(old,other,residual,rate,lambda)
 Float16(muladd(rate,muladd(residual,other,-lambda*old),old))
end
@inline function body_sgd!(gid,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 lane=rem(gid,Int32(32));worker=div(gid,Int32(32));mask=UInt32(0xffffffff)
 # Source __ldg fields are preserved with exported CUDA.jl cached scalar loads.
 # A struct load would otherwise use the generic non-LDG fallback.
 words=reinterpret(Int32,R)
 @inbounds for epoch=current_iter:current_iter+num_iters-Int32(1)
  rate=CUDA.ldg(rates,epoch+Int32(1))
  for group=Int32(0):updates-Int32(1)
   start=Int64(0)
   if lane==0
    start=rem(unsafe_trunc(Int64,source_random!(state,worker+Int32(1))*Float32(nnz)),nnz)
   end
   start=CUDA.shfl_sync(mask,start,Int32(0))
   if lane==0
    slot=((epoch-current_iter)*updates+group)*Int32(length(state))+worker+Int32(1)
    trace[slot]=start
   end
   for step=Int32(0):vector_size-Int32(1)
    offset=Int32(rem(start+Int64(step),nnz));at=offset*Int32(3)
    u=CUDA.ldg(words,at+Int32(1));v=CUDA.ldg(words,at+Int32(2));rating=reinterpret(Float32,CUDA.ldg(words,at+Int32(3)))
    bp=u*k+lane+Int32(1);bq=v*k+lane+Int32(1)
    p1=Float32(p[bp]);q1=Float32(q[bq]);p2=Float32(p[bp+Int32(32)]);q2=Float32(q[bq+Int32(32)])
    p3=Float32(p[bp+Int32(64)]);q3=Float32(q[bq+Int32(64)]);p4=Float32(p[bp+Int32(96)]);q4=Float32(q[bq+Int32(96)])
    product=muladd(p4,q4,muladd(p3,q3,muladd(p2,q2,p1*q1)))
    product+=CUDA.shfl_down_sync(mask,product,Int32(16));product+=CUDA.shfl_down_sync(mask,product,Int32(8))
    product+=CUDA.shfl_down_sync(mask,product,Int32(4));product+=CUDA.shfl_down_sync(mask,product,Int32(2));product+=CUDA.shfl_down_sync(mask,product,Int32(1))
    product=CUDA.shfl_sync(mask,product,Int32(0));residual=rating-product
    p[bp]=source_update(p1,q1,residual,rate,lambda_p);q[bq]=source_update(q1,p1,residual,rate,lambda_q)
    p[bp+Int32(32)]=source_update(p2,q2,residual,rate,lambda_p);q[bq+Int32(32)]=source_update(q2,p2,residual,rate,lambda_q)
    p[bp+Int32(64)]=source_update(p3,q3,residual,rate,lambda_p);q[bq+Int32(64)]=source_update(q3,p3,residual,rate,lambda_q)
    p[bp+Int32(96)]=source_update(p4,q4,residual,rate,lambda_p);q[bq+Int32(96)]=source_update(q4,p4,residual,rate,lambda_q)
   end
  end
 end;nothing
end
