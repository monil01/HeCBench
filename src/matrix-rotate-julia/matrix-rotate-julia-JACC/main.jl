using CUDA, JACC
include(joinpath(@__DIR__, "..", "..", "_portable_study", "OwnedStudy.jl"))
function rotate_jacc!(index,matrix,n::Int32)
    layer=index-1
    if layer < n÷2
        first=layer; last=n-1-layer
        for i in first:last-1
            off=i-first
            top=first*n+i+1; left=(last-off)*n+first+1
            bottom=last*n+last-off+1; right=i*n+last+1
            value=matrix[top]
            matrix[top]=matrix[left]; matrix[left]=matrix[bottom]
            matrix[bottom]=matrix[right]; matrix[right]=value
        end
    end
end
function main()
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1024
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    n>=2 && n*n<=2^24 || error("fixture requires n>=2 and exactly represented Float32 indices")
    host=Float32.(0:n*n-1); input=reshape(host,n,n)
    # Source stores row-major: clockwise row-major rotation equals this
    # transpose-aware column-major mapping, applied repeats modulo four times.
    expected=copy(input)
    for _ in 1:mod(repeats,4)
        expected=reverse(permutedims(expected),dims=1)
    end
    device=CuArray(host); spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    launch!()=JACC.parallel_for(spec,n÷2,rotate_jacc!,device,Int32(n))
    OwnedStudy.run_trials!(launch!,()->copyto!(device,host),()->Array(device),vec(expected),repeats)
end
main()
