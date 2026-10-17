using CUDA,Printf
function quant3_kernel!(vec,mat,out,scales,zeros,width::Int32)
    tid=threadIdx().x
    group=blockIdx().x-Int32(1)
    col=(blockIdx().y-Int32(1))*Int32(256)+tid
    tile=CuStaticSharedArray(Float32,256)
    @inbounds tile[tid]=vec[group*Int32(256)+tid]
    sync_threads()
    acc=0f0
    @inbounds for r in Int32(0):Int32(255)
        bit=r*Int32(3);word=div(bit,Int32(32));shift=rem(bit,Int32(32))
        packed=reinterpret(UInt32,mat[(group*Int32(24)+word)*width+col])
        q=(packed>>shift)&UInt32(7)
        if shift>Int32(29)
            nextword=reinterpret(UInt32,mat[(group*Int32(24)+word+Int32(1))*width+col])
            q=((packed>>shift)|(nextword<<(Int32(32)-shift)))&UInt32(7)
        end
        acc+=(scales[col]*Float32(q)-zeros[col])*tile[r+Int32(1)]
    end
    CUDA.@atomic out[col]+=acc
    return
end
function main(args)
    length(args)==1 || error("Usage: main.jl <repeat>")
    repeats=parse(Int,args[1]);m=49152;n=12288;height=div(m,32)*3
    vec=CUDA.ones(Float32,m);packed=Vector{UInt32}(undef,height*n);p1=UInt32(0b01001001001001001001001001001001)
    for row in 0:height-1,col in 1:n;packed[row*n+col]=p1<<mod(row,3);end
    mat=CuArray(collect(reinterpret(Int32,packed)));scales=CUDA.fill(-1f0,n);zeros=CUDA.zeros(Float32,n);out=CUDA.zeros(Float32,n)
    launch! = ()->begin
        fill!(out,0f0)
        @cuda threads=256 blocks=(div(m,256),div(n,256)) quant3_kernel!(vec,mat,out,scales,zeros,Int32(n))
    end
    launch!();CUDA.synchronize();start=time_ns()
    for _ in 1:repeats;launch!();end
    CUDA.synchronize();elapsed=(time_ns()-start)*1e-3/repeats
    @printf("Average execution time of packed3-bit Float32 kernel: %f (us)\n",elapsed)
    ok=all(Array(out).==-Float32(m));println(ok ? "PASS" : "FAIL");return ok ? 0 : 1
end
exit(main(ARGS))
