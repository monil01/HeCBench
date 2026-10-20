# Direct Julia translations of reference_history/hash_table.h and dUtils.cuh.
# C literals: node0=true, literal0=true and literal1=false.
@inline lr_s_AigNodeID(x)=x>>1
@inline lr_s_AigNodeIsComplement(x)=!iszero(x&Int32(1))
@inline lr_s_AigNodeLitCond(x,c)=(x<<1)+Int32(c)
@inline lr_s_AigNodeNotCond(x,c)=xor(x,Int32(c))
@inline lr_s_formAndNodeKey(a,b)=(UInt64(lr_cast(UInt32,a))<<32)|UInt64(lr_cast(UInt32,b))
@inline function lr_s_unbindAndNodeKeys(key,a,b)
    b[0]=lr_cast(UInt32,key&UInt64(0xffffffff));a[0]=lr_cast(UInt32,key>>32);return nothing
end
@inline function lr_s_checkTrivialAndCases(a,b)::UInt32
    a==b && return lr_cast(UInt32,a)
    a==xor(b,Int32(1)) && return UInt32(1)
    a==0 && return lr_cast(UInt32,b)
    b==0 && return lr_cast(UInt32,a)
    (a==1||b==1) && return UInt32(1)
    return typemax(UInt32)
end
@inline function lr_s_hash(x::UInt64)
    x=xor(x,x>>33);x*=UInt64(0xff51afd7ed558ccd)
    x=xor(x,x>>33);x*=UInt64(0xc4ceb9fe1a85ec53)
    return xor(x,x>>33)
end
@inline function lr_s_hash(x::UInt32)
    x=xor(x,x>>16);x*=UInt32(0x85ebca6b)
    x=xor(x,x>>13);x*=UInt32(0xc2b2ae35)
    return xor(x,x>>16)
end
@inline function lr_s_retrieve_single(keys::LRPtr{K},values::LRPtr{V},key::K,capacity)::V where {K,V}
    loc=rem(lr_s_hash(key),K(capacity))
    while true
        keys[loc]==key && return values[loc]
        keys[loc]==typemax(K) && return typemax(V)
        loc=rem(loc+K(1),K(capacity))
    end
end
@inline function lr_s_insert_single_no_update(keys::LRPtr{K},values::LRPtr{V},key::K,value::V,capacity)::UInt32 where {K,V}
    loc=rem(lr_s_hash(key),K(capacity))
    while true
        prev=lr_atomicCAS(keys+loc,typemax(K),key)
        if prev==typemax(K)
            values[loc]=value;return UInt32(2)
        elseif prev==key
            return lr_cast(UInt32,values[loc])
        end
        loc=rem(loc+K(1),K(capacity))
    end
end
@inline lr_atomicSub(p::LRPtr{T},v) where T = CUDA.atomic_sub!(lr_atomic_pointer(p),lr_cast(T,v))
# Direct source getEntryIndicator/gatherKeyValues device bodies.
@inline function lr_s_getEntryIndicator(ctx,indicator,keys::LRPtr{K},capacity) where K
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<capacity;indicator[idx]=Int32(keys[idx]!=typemax(K));end
    return nothing
end
@inline function lr_s_gatherKeyValues(ctx,scanned,sourcekeys,sourcevalues,keys,values,capacity)
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx==0
        if scanned[0]==1;keys[0]=sourcekeys[0];values[0]=sourcevalues[0];end
    elseif idx<capacity
        if scanned[idx]>scanned[idx-Int32(1)]
            dest=scanned[idx]-Int32(1);keys[dest]=sourcekeys[idx];values[dest]=sourcevalues[idx]
        end
    end
    return nothing
end
# Original header batch kernels: duplicate insert returns without reading value.
@inline function lr_s_insert_batch_entry(keys::LRPtr{K},values::LRPtr{V},key::K,value::V,capacity)::Nothing where {K,V}
    loc=rem(lr_s_hash(key),K(capacity))
    while true
        prev=lr_atomicCAS(keys+loc,typemax(K),key)
        if prev==typemax(K);values[loc]=value;return nothing;end
        prev==key && return nothing
        loc=rem(loc+K(1),K(capacity))
    end
end
@inline function lr_s_insert_batch_no_update_kernel(ctx,hk,hv,keys,values,len,capacity)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<len;lr_s_insert_batch_entry(hk,hv,keys[idx],values[idx],capacity);end
    return nothing
end
@inline function lr_s_insert_batch_no_update_masked_kernel(ctx,hk,hv,keys,values,mask,len,capacity)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<len&&mask[idx]==1;lr_s_insert_batch_entry(hk,hv,keys[idx],values[idx],capacity);end
    return nothing
end
@inline function lr_s_retrieve_batch_kernel(ctx,hk,hv,keys,values,len,capacity)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<len;values[idx]=lr_s_retrieve_single(hk,hv,keys[idx],capacity);end
    return nothing
end
@inline function lr_s_retrieve_batch_masked_kernel(ctx,hk,hv,keys,values,mask,len,capacity)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<len&&mask[idx]==1;values[idx]=lr_s_retrieve_single(hk,hv,keys[idx],capacity);end
    return nothing
end
