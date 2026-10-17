using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function generate_ka!(table,table_size,n)
    i=Int32(@index(Global,Linear))-Int32(1)
    if i<table_size
        @inbounds table[i+1]=i<n ? i : Int32(-1)
    end
end
@kernel function sort_ka!(table,table_size,j,k,sequence,n)
    i=Int32(@index(Global,Linear))-Int32(1)
    partner=xor(i,j)
    if i<table_size && partner>i
        ascending=(i&k)==0
        @inbounds a=table[i+1];b=table[partner+1]
        if rotation_less(ascending ? b : a,ascending ? a : b,sequence,n)
            @inbounds table[i+1]=b;table[partner+1]=a
        end
    end
end
@kernel function reconstruct_ka!(table,sequence,output,n)
    i=Int32(@index(Global,Linear))-Int32(1)
    if i<n
        @inbounds output[i+1]=sequence[(n+table[i+1]-Int32(1))%n+Int32(1)]
    end
end
function factory(input,table,output,n,table_size)
    gen=generate_ka!(CUDABackend(),256);sort=sort_ka!(CUDABackend(),256);rec=reconstruct_ka!(CUDABackend(),256)
    ()->begin
        gen(table,table_size,n;ndrange=Int(table_size))
        k=Int32(2)
        while k<=table_size
            j=k>>1
            while j>0
                sort(table,table_size,j,k,input,n;ndrange=Int(table_size))
                j>>=1
            end
            k<<=1
        end
        rec(table,input,output,n;ndrange=Int(n))
    end
end
bwt_study(factory)
