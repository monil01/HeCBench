@inline function dot_partial(ref,q,tx,block,blocks,n)
    tid=block*Int32(128)+tx-1;temp=Int32(0)
    while tid<n
        @inbounds temp+=ref[tid+17]*q[tid+17]
        tid+=Int32(128)*blocks
    end
    temp
end
@inline function count_partial(scores,threshold,tx,block,blocks,n)
    tid=block*Int32(128)+tx-1;temp=Int32(0)
    while tid<n
        @inbounds temp+=scores[tid+17]>threshold ? Int32(1) : Int32(0)
        tid+=Int32(128)*blocks
    end
    temp
end
@inline function random_score!(idx,random,ref,scores,m,uc,set,gens)
    if idx<gens
        temp=0.0f0;col=idx
        while col<m
            @inbounds n=1.0f0-random[col+17]
            factor=1.0f0/Float32(set)
            if n>=0.5f0
                factor=-factor;n-=0.5f0
            end
            gene=floor(Int32,(n*22283.0f0)*2.0f0)
            @inbounds temp=muladd(Float32(ref[gene+17]),factor,temp)
            col+=gens
        end
        @inbounds scores[idx+17]=temp/uc
    end
    return nothing
end
