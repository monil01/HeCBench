# Source balance helpers and direct translation of its device Thrust shuffle.
# Source Thrust sequential sort is stable; insertion sort below preserves its
# observable order, with O(length²) rather than merge-sort work disclosed.
@inline lr_s_AigIsNode(x,pi)=Int32(x>pi)
@inline lr_s_AigIsPIConst(x,pi)=Int32(x<=pi)
@inline lr_s_AigNodeNot(x)=xor(x,Int32(1))
@inline function lr_balance_permute!(arrays,levels,start,len,seed)
    keys=lr_alloc(UInt32,24);tmpa=lr_alloc(Int32,len);tmpl=lr_alloc(Int32,len)
    state=rem(UInt64(seed),UInt64(2147483647));state==0&&(state=UInt64(1))
    for r in Int32(0):Int32(23)
        state=rem(UInt64(48271)*state,UInt64(2147483647));keys[r]=UInt32(state)
    end
    bits=Int32(4)
    if len>16
        bits=Int32(0);m=UInt64(len-1)
        while m!=0;bits+=Int32(1);m>>=1;end
    end
    lb=bits÷Int32(2);rb=bits-lb;lm=(UInt32(1)<<lb)-UInt32(1);rm=(UInt32(1)<<rb)-UInt32(1)
    dest=Int32(0)
    for value in UInt64(0):(UInt64(1)<<bits)-UInt64(1)
        left=UInt32(value>>rb);right=UInt32(value&UInt64(rm))
        for r in Int32(0):Int32(23)
            product=UInt64(0xd2b74407b1ce6e93)*UInt64(left)
            hi=lr_cast(UInt32,product>>32);lo=lr_cast(UInt32,product)
            lo=(lo<<(rb-lb))|(right>>lb)
            left=xor(xor(hi,keys[r]),right)&lm;right=lo&rm
        end
        q=UInt64(left<<rb)|UInt64(right)
        if q<UInt64(len)
            tmpa[dest]=arrays[start+Int32(q)];tmpl[dest]=levels[start+Int32(q)];dest+=Int32(1)
        end
    end
    CUDA.@cuassert dest==len
    for j in Int32(0):len-Int32(1);arrays[start+j]=tmpa[j];levels[start+j]=tmpl[j];end
    lr_free(reinterpret(Ptr{Cvoid},getfield(keys,:base)));lr_free(reinterpret(Ptr{Cvoid},getfield(tmpa,:base)));lr_free(reinterpret(Ptr{Cvoid},getfield(tmpl,:base)));return nothing
end
@inline function lr_prepareReconstructArrays(ctx,gathered,nodeCover,coverNode,newLit,table,links,lens,levelkeys,levelvalues,capacity,arrays,levels,lengths,nready,npi,maxlen,sortDecId)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<nready
        old=gathered[idx];row=old;column=Int32(0);len=lens[old];start=idx*maxlen
        CUDA.@cuassert len<=maxlen
        lengths[idx]=len
        len==0 && return nothing
        for j in Int32(0):len-Int32(1)
            if column==4;column=Int32(0);row=links[row];end
            lit=table[row*Int32(4)+column];column+=Int32(1);id=lit>>1
            if id<=npi
                arrays[start+j]=lit;levels[start+j]=Int32(0)
            else
                q=newLit[nodeCover[id]];arrays[start+j]=xor(q,lit&Int32(1));qid=q>>1
                if qid<=npi
                    levels[start+j]=Int32(0)
                else
                    lev=lr_s_retrieve_single(levelkeys,levelvalues,UInt32(qid),capacity)
                    CUDA.@cuassert lev!=typemax(UInt32)
                    levels[start+j]=lr_cast(Int32,lev)
                end
            end
        end
        if sortDecId==0;lr_balance_permute!(arrays,levels,start,len,idx);end
        # Stable order by decreasing level, optionally decreasing literal on ties.
        for j in Int32(1):len-Int32(1)
            literal=arrays[start+j];level=levels[start+j];k=j-Int32(1)
            while k>=0&&(levels[start+k]<level||(sortDecId!=0&&levels[start+k]==level&&arrays[start+k]<literal))
                arrays[start+k+Int32(1)]=arrays[start+k];levels[start+k+Int32(1)]=levels[start+k];k-=Int32(1)
            end
            arrays[start+k+Int32(1)]=literal;levels[start+k+Int32(1)]=level
        end
    end
    return nothing
end
