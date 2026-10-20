# Original Pangborn GMM mathematical bodies translated to Julia; license retained.
using CUDA,StaticArrays
@inline gmm_at(a,i)=@inbounds a[Int32(i)+Int32(1)]
@inline function gmm_put!(a,i,x)
    @inbounds a[Int32(i)+Int32(1)]=Float32(x)
    nothing
end
@inline function gmm_invert!(a,n,detarg)
    # The source's sequential in-place LU inversion, not a vendor replacement.
    det=0f0
    if n==1
        det=log(gmm_at(a,0));gmm_put!(a,0,1f0/gmm_at(a,0))
    else
        for i in Int32(1):n-Int32(1);gmm_put!(a,i,gmm_at(a,i)/gmm_at(a,0));end
        for i in Int32(1):n-Int32(1)
            for j in i:n-Int32(1)
                sum=0f0
                for k in Int32(0):i-Int32(1);sum=muladd(gmm_at(a,j*n+k),gmm_at(a,k*n+i),sum);end
                gmm_put!(a,j*n+i,gmm_at(a,j*n+i)-sum)
            end
            if i==n-Int32(1);continue;end
            for j in i+Int32(1):n-Int32(1)
                sum=0f0
                for k in Int32(0):i-Int32(1);sum=muladd(gmm_at(a,i*n+k),gmm_at(a,k*n+j),sum);end
                gmm_put!(a,i*n+j,(gmm_at(a,i*n+j)-sum)/gmm_at(a,i*n+i))
            end
        end
        for i in Int32(0):n-Int32(1);det+=log(abs(gmm_at(a,i*n+i)));end
        for i in Int32(0):n-Int32(1),j in i:n-Int32(1)
            x=1f0
            if i!=j
                x=0f0
                for k in i:j-Int32(1);x=muladd(-gmm_at(a,j*n+k),gmm_at(a,k*n+i),x);end
            end
            gmm_put!(a,j*n+i,x/gmm_at(a,j*n+j))
        end
        for i in Int32(0):n-Int32(1),j in i:n-Int32(1)
            if i==j;continue;end
            sum=0f0
            for k in i:j-Int32(1);sum=muladd(gmm_at(a,k*n+j),i==k ? 1f0 : gmm_at(a,i*n+k),sum);end
            gmm_put!(a,i*n+j,-sum)
        end
        for i in Int32(0):n-Int32(1),j in Int32(0):n-Int32(1)
            sum=0f0
            for k in max(i,j):n-Int32(1);sum=muladd(j==k ? 1f0 : gmm_at(a,j*n+k),gmm_at(a,k*n+i),sum);end
            gmm_put!(a,j*n+i,sum)
        end
    end
    @inbounds detarg[1]=det
    nothing
end
@inline function gmm_xorsum!(buf,tid,offset)
    CUDA.sync_threads()
    bit=Int32(128)
    while bit>0
        value=(@inbounds buf[offset+tid+Int32(1)])+(@inbounds buf[offset+xor(tid,bit)+Int32(1)])
        CUDA.sync_threads()
        @inbounds buf[offset+tid+Int32(1)]=value
        CUDA.sync_threads();bit=bit>>1
    end
    @inbounds buf[offset+tid+Int32(1)]
end
@inline function gmm_cell2(index,d)
    i=Int32(0)
    for row in Int32(0):d-Int32(1),col in Int32(0):row
        if i==index;return row,col;end
        i+=Int32(1)
    end
    return Int32(0),Int32(0)
end
