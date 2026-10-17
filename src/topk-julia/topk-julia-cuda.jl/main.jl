using CUDA,Printf
function topk_permutation_kernel!(input,ids,values,batch::Int32,hidden::Int32,k::Int32,total::Int32)
    index=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    if index<=total
        zero=index-Int32(1);col=rem(zero,hidden);row=div(zero,hidden)
        value=@inbounds input[index];integer=Int32(value)
        if integer>=hidden-k && value==Float32(integer)
            rank=hidden-Int32(1)-integer;out=row*k+rank+Int32(1)
            @inbounds ids[out]=col
            @inbounds values[out]=value
        end
    end
    return
end
function main(args)
    length(args)==2 || error("Usage: main.jl <batch_size> <repeat>")
    batch=parse(Int,args[1]);repeats=parse(Int,args[2])
    for hidden in (3072,4096,8192,16384,32768,65536,131072),k in (2048,1024)
        host=repeat(Float32.(0:hidden-1),batch);input=CuArray(host);ids=CUDA.zeros(Int32,batch*k);values=CUDA.zeros(Float32,batch*k)
        launch! = ()->(@cuda threads=256 blocks=cld(batch*hidden,256) topk_permutation_kernel!(input,ids,values,Int32(batch),Int32(hidden),Int32(k),Int32(batch*hidden)))
        launch!();CUDA.synchronize();start=time_ns()
        for _ in 1:repeats;launch!();end
        CUDA.synchronize();elapsed=(time_ns()-start)*1e-3/repeats
        got=Array(values);gotids=Array(ids);ok=true
        for row in 0:batch-1,rank in 0:k-1
            value=hidden-1-rank;ok &= got[row*k+rank+1]==Float32(value)&&gotids[row*k+rank+1]==value
        end
        @printf("Average execution time of topk : %f (us)\n",elapsed);println(ok ? "PASS" : "FAIL")
        ok || return 1
    end
    return 0
end
exit(main(ARGS))
