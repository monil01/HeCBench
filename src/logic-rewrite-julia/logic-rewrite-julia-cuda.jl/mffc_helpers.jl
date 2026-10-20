# Literal/hash composition from the unchanged strash.cuh helper.
@inline function lr_s_retrieveHashTableCheckTrivial(a,b,keys,values,capacity)
    v=lr_s_checkTrivialAndCases(a,b)
    v!=typemax(UInt32) && return v
    return lr_s_retrieve_single(keys,values,lr_s_formAndNodeKey(a,b),capacity)
end
# Source Table::gatherTableToConsecutive<int,16>.
@inline function lr_gatherTableToConsecutive(ctx,table,lens,indices,ranges,out,n)::Nothing
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    if idx<n
        row=indices[idx];start=idx==0 ? Int32(0) : ranges[idx-Int32(1)]
        stop=ranges[idx];CUDA.@cuassert stop-start==lens[row]
        for j in Int32(0):lens[row]-Int32(1);out[start+j]=table[row*Int32(16)+j];end
    end
    return nothing
end
# Source getCutTruthTableConsecutive<192>: identical per-lane traversal,
# 32-lane exclusive scan, one heap allocation per warp, and warp completion.
# CUB's scan is expressed with CUDA shuffle intrinsics. Julia uses128B static
# shared pointers; source additionally declares CUB TempStorage. Traversal arrays
# use explicit device heap rather than C++ automatic local storage (covariate).
@inline function lr_getCutTruthTableConsecutive(ctx,a,b,saved,indices,cuts,ranges,truth,truthranges,elem,n,npi,maxcut,node2cone)::Nothing
    shared=CUDA.CuStaticSharedArray(UInt64,16)
    idx=Int32(blockIdx().x-1)*Int32(blockDim().x)+Int32(threadIdx().x-1)
    lane=Int32(threadIdx().x-1)&Int32(31);warp=Int32(threadIdx().x-1)÷Int32(32)
    nthreads=((n+Int32(31))÷Int32(32))*Int32(32)
    CUDA.@cuassert n<=nthreads&&nthreads-n<32
    if idx<nthreads
        stack=lr_alloc(Int32,192);result=lr_alloc(Int32,192);visited=lr_alloc(Int32,192);vs=lr_alloc(Int32)
        vars=Int32(0);ninternal=Int32(0);words=Int32(0);start=Int32(0);root=Int32(0);rtop=Int32(-1)
        if idx<n
            root=indices[idx];start=idx==0 ? Int32(0) : ranges[idx-Int32(1)];stop=ranges[idx];vars=stop-start;words=lr_factor_TruthWordNum(vars)
            vsize=Int32(0)
            for j in Int32(0):vars-Int32(1);visited[vsize]=cuts[start+j];vsize+=Int32(1);end
            top=Int32(0);stack[top]=root
            while top!=-1
                node=stack[top];top-=Int32(1);seen=false
                for j in Int32(0):vsize-Int32(1)
                    if visited[j]==node;seen=true;break;end
                end
                seen && continue
                CUDA.@cuassert node>npi&&vsize<192
                visited[vsize]=node;vsize+=Int32(1)
                if lr_truth(node2cone)&&node!=root;node2cone[node]=idx;end
                j=rtop;rtop+=Int32(1)
                while j>=0&&result[j]<node;result[j+Int32(1)]=result[j];j-=Int32(1);end
                result[j+Int32(1)]=node
                CUDA.@cuassert top<190
                top+=Int32(1);stack[top]=b[node]>>1;top+=Int32(1);stack[top]=a[node]>>1
            end
            ninternal=rtop+Int32(1)
            CUDA.@cuassert ninternal>=saved[root]&&ninternal+vars<192&&result[0]==root
        end
        localwords=idx<n ? (vars+ninternal)*words : Int32(0);scan=localwords
        for delta in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
            q=CUDA.shfl_up_sync(UInt32(0xffffffff),scan,delta)
            if lane>=delta;scan+=q;end
        end
        total=CUDA.shfl_sync(UInt32(0xffffffff),scan,32);offset=scan-localwords
        if lane==0
            ptr=CUDA.malloc(Csize_t(total)*Csize_t(4));CUDA.@cuassert ptr!=C_NULL;shared[warp+Int32(1)]=UInt64(UInt(ptr))
        end
        CUDA.sync_warp(UInt32(0xffffffff));mem=lr_ptr(UInt32,shared[warp+Int32(1)])+offset
        if idx<n
            vs[0]=Int32(0);elementwords=lr_factor_TruthWordNum(maxcut)
            for j in Int32(0):vars-Int32(1)
                for w in Int32(0):words-Int32(1);mem[j*words+w]=elem[j*elementwords+w];end
                visited[vs[0]]=cuts[start+j];vs[0]+=Int32(1)
            end
            for j in rtop:Int32(-1):Int32(0);lr_cutTruthIter(ctx,result[j],a,b,mem,visited,vs,words);end
            CUDA.@cuassert visited[vs[0]-Int32(1)]==root
            starttruth=idx==0 ? Int32(0) : truthranges[idx-Int32(1)]
            CUDA.@cuassert truthranges[idx]-starttruth==words
            for w in Int32(0):words-Int32(1);truth[starttruth+w]=mem[(vs[0]-Int32(1))*words+w];end
        end
        CUDA.sync_warp(UInt32(0xffffffff))
        if lane==0;lr_free(Ptr{Cvoid}(shared[warp+Int32(1)]));end
        for cell in (stack,result,visited,vs);lr_free(Ptr{Cvoid}(getfield(cell,:base)));end
    end
    return nothing
end
