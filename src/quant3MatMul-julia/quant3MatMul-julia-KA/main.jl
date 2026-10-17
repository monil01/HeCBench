using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function quant3_ka!(vec,mat,out,scales,zeros,width::Int32)
    tid=Int32(@index(Local,Linear))
    group=Int32(@index(Group,NTuple)[1])-Int32(1)
    col=(Int32(@index(Group,NTuple)[2])-Int32(1))*Int32(256)+tid
    tile=@localmem Float32 (256,)
    @inbounds tile[tid]=vec[group*Int32(256)+tid]
    @synchronize
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
    Atomix.@atomic out[col]+=acc
end
function factory(vec,mat,out,scales,zeros)
    kernel=quant3_ka!(get_backend(vec),(256,1))
    return ()->kernel(vec,mat,out,scales,zeros,Int32(1024);ndrange=(16*256,4))
end
quant3_study(factory)
