using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function generate_cuda!(table,table_size,n)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    if i<table_size
        @inbounds table[i+1]=i<n ? i : Int32(-1)
    end
    return
end
function sort_cuda!(table,table_size,j,k,sequence,n)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    partner=xor(i,j)
    if i<table_size && partner>i
        ascending=(i&k)==0
        @inbounds a=table[i+1];b=table[partner+1]
        if rotation_less(ascending ? b : a,ascending ? a : b,sequence,n)
            @inbounds table[i+1]=b;table[partner+1]=a
        end
    end
    return
end
function reconstruct_cuda!(table,sequence,output,n)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    if i<n
        @inbounds output[i+1]=sequence[(n+table[i+1]-Int32(1))%n+Int32(1)]
    end
    return
end
function factory(input,table,output,n,table_size)
    ()->begin
        @cuda threads=256 blocks=4 generate_cuda!(table,table_size,n)
        k=Int32(2)
        while k<=table_size
            j=k>>1
            while j>0
                @cuda threads=256 blocks=4 sort_cuda!(table,table_size,j,k,input,n)
                j>>=1
            end
            k<<=1
        end
        @cuda threads=256 blocks=4 reconstruct_cuda!(table,input,output,n)
    end
end
bwt_study(factory)
