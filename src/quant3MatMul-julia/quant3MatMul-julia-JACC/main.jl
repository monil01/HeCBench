using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function quant3_jacc!(gx,gy,vec,mat,out,scales,zeros,width::Int32)
    tid=Int32(mod(gx-1,256)+1)
    group=Int32(div(gx-1,256))
    col=(Int32(gy)-Int32(1))*Int32(256)+tid
    tile=JACC.shared(@view vec[group*Int32(256)+1:group*Int32(256)+256])
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
    JACC.@atomic out[col]+=acc
    return
end
function factory(vec,mat,out,scales,zeros)
    spec=JACC.launch_spec(;threads=(256,1),shmem_size=1024,sync=false)
    return ()->JACC.parallel_for(spec,(16*256,4),quant3_jacc!,vec,mat,out,scales,zeros,Int32(1024))
end
quant3_study(factory)
