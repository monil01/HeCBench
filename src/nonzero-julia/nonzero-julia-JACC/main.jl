using CUDA,JACC
function prefix_jacc!(i,input,prefix,totals,n)
    base=((i-1)÷256)*256;acc=Int32(0)
    for k in base:i-2;@inbounds acc+=Int32(input[k+1]!=0);end
    @inbounds prefix[i]=acc
    if i%256==0||i==n;@inbounds totals[(i-1)÷256+1]=acc+Int32(input[i]!=0);end
    return nothing
end
function offsets_jacc!(b,totals,offsets,count,nb)
    acc=Int32(0)
    for k in 1:b-1;@inbounds acc+=totals[k];end
    @inbounds offsets[b]=acc
    if b==nb;@inbounds count[1]=Int64(acc+totals[b]);end
    return nothing
end
function scatter_jacc!(i,input,out,prefix,offsets,n,nc,nz)
    @inbounds if input[i]!=0
        k=Int64(offsets[(i-1)÷256+1]+prefix[i])+1
        out[k]=Int64(i-1)÷nc;out[nz+k]=Int64(i-1)%nc
    end
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(input,out,prefix,totals,offsets,count,n,nb,nr,nc)
    JACC.parallel_for(SPEC,Int(n),prefix_jacc!,input,prefix,totals,n)
    JACC.parallel_for(SPEC,Int(nb),offsets_jacc!,totals,offsets,count,nb)
    nz=Array(count)[1]
    JACC.parallel_for(SPEC,Int(n),scatter_jacc!,input,out,prefix,offsets,n,nc,nz)
    return nz
end
include(joinpath(@__DIR__,"..","study_common.jl"));nonzero_study(launch_jacc!)
