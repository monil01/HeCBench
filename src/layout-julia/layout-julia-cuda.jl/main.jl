using CUDA,Printf
# Faithful correction: original_main.jl synthesized results from indices and
# omitted input memory. These kernels load the actual AoS/SoA buffers.
const TREE_NUM=4096
const TREE_SIZE=4096
const GROUP_SIZE=256
function aos_kernel!(input,out)
    gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
    if gid<TREE_NUM
        value=UInt32(0)
        for k in 0:TREE_SIZE-1
            value+=reinterpret(UInt32,input[gid*TREE_SIZE+k+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
    return
end
function soa_kernel!(input,out)
    gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
    if gid<TREE_NUM
        value=UInt32(0)
        for k in 0:TREE_SIZE-1
            value+=reinterpret(UInt32,input[k*TREE_NUM+gid+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
    return
end
function main(args=ARGS)
    repeats=length(args)>0 ? parse(Int,args[1]) : 10
    host=Int32.(0:TREE_NUM*TREE_SIZE-1);expected=Int32[reinterpret(Int32,UInt32(mod(sum(Int64,view(host,r*TREE_SIZE+1:(r+1)*TREE_SIZE)),Int64(1)<<32))) for r in 0:TREE_NUM-1]
    for layout in ("aos","soa")
        data=layout=="aos" ? host : vec(permutedims(reshape(host,TREE_SIZE,TREE_NUM)))
        input=CuArray(data);out=CUDA.zeros(Int32,TREE_NUM)
        launch! = layout=="aos" ? ()->(@cuda threads=GROUP_SIZE blocks=cld(TREE_NUM,GROUP_SIZE) aos_kernel!(input,out)) : ()->(@cuda threads=GROUP_SIZE blocks=cld(TREE_NUM,GROUP_SIZE) soa_kernel!(input,out))
        launch!();CUDA.synchronize();start=time_ns();for _ in 1:repeats;launch!();end;CUDA.synchronize()
        @printf("Average kernel execution time (%s): %.8g (us)\n",uppercase(layout),(time_ns()-start)*1e-3/repeats)
        Array(out)==expected || error("$layout numerical mismatch")
        println("PASS")
    end
end
if get(ENV,"HECBENCH_STUDY_INCLUDE_ONLY","0")!="1"
    main()
end
