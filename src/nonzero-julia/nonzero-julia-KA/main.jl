using CUDA,KernelAbstractions
@kernel function prefix_ka!(input,prefix,totals,n)
    i=@index(Global,Linear)
    if i<=n
        base=((i-1)÷256)*256;acc=Int32(0)
        for k in base:i-2;@inbounds acc+=Int32(input[k+1]!=0);end
        @inbounds prefix[i]=acc
        if i%256==0||i==n;@inbounds totals[(i-1)÷256+1]=acc+Int32(input[i]!=0);end
    end
end
@kernel function offsets_ka!(totals,offsets,count,nb)
    b=@index(Global,Linear)
    if b<=nb
        acc=Int32(0)
        for j in 1:b-1;@inbounds acc+=totals[j];end
        @inbounds offsets[b]=acc
        if b==nb;@inbounds count[1]=Int64(acc+totals[b]);end
    end
end
@kernel function scatter_ka!(input,out,prefix,offsets,n,nc,nz)
    i=@index(Global,Linear)
    if i<=n
        @inbounds if input[i]!=0
            j=Int64(offsets[(i-1)÷256+1]+prefix[i])+1
            out[j]=Int64(i-1)÷nc;out[nz+j]=Int64(i-1)%nc
        end
    end
end
function launch_ka!(input,out,prefix,totals,offsets,count,n,nb,nr,nc)
    backend=get_backend(input)
    prefix_ka!(backend,256)(input,prefix,totals,n;ndrange=Int(n))
    offsets_ka!(backend,256)(totals,offsets,count,nb;ndrange=Int(nb))
    nz=Array(count)[1]
    scatter_ka!(backend,256)(input,out,prefix,offsets,n,nc,nz;ndrange=Int(n))
    return nz
end
include(joinpath(@__DIR__,"..","study_common.jl"));nonzero_study(launch_ka!)
