# Original graph-manager source device bodies.
@inline function lr_processRwmanFanins(ctx,a,b,refs,npi,n)
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<n
        node=idx+npi+Int32(1)
        if a[node]<2;a[node]=Int32(1)-a[node];end
        if b[node]<2;b[node]=Int32(1)-b[node];end
        lr_atomicAdd(refs+(a[node]>>1),Int32(1));lr_atomicAdd(refs+(b[node]>>1),Int32(1))
    end
    return nothing
end
@inline function lr_processRwmanOuts(ctx,outs,refs,n)
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<n
        if outs[idx]<2;outs[idx]=Int32(1)-outs[idx];end
        lr_atomicAdd(refs+(outs[idx]>>1),Int32(1))
    end
    return nothing
end
@inline function lr_updateDeviceStats(ctx,n,npi,pn,pobjects)
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx==0;pn[0]=n;pobjects[0]=n+npi+Int32(1);end
    return nothing
end
@inline function lr_printStatsKernel(ctx,np,no,nn)
    CUDA.@cuprintf("AIG stats: i/o = %d/%d and = %d",np[0],no[0],nn[0]);return nothing
end
