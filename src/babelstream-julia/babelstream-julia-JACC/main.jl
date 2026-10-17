using CUDA,JACC
function stream_map!(i,a,b,c,op)
    @inbounds if op==1
        c[i]=a[i]
    elseif op==2
        b[i]=0.4f0*c[i]
    elseif op==3
        c[i]=a[i]+b[i]
    elseif op==4
        a[i]=b[i]+0.4f0*c[i]
    elseif op==5
        a[i]+=b[i]+0.4f0*c[i]
    end
    return
end
function stream_dot!(i,a,b,partial,n,template)
    lid=(i-1)%256+1;group=(i-1)÷256+1
    scratch=JACC.shared(template)
    acc=0f0;j=i
    @inbounds while j<=n
        acc+=a[j]*b[j];j+=65536
    end
    scratch[lid]=acc
    offset=128
    while offset>0
        JACC.sync_workgroup()
        if lid<=offset
            @inbounds scratch[lid]+=scratch[lid+offset]
        end
        offset÷=2
    end
    if lid==1
        @inbounds partial[group]=scratch[1]
    end
    return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(op,a,b,c,partial,n)
    if op=="dot"
        template=CUDA.zeros(Float32,256)
        spec=JACC.launch_spec(;threads=256,blocks=256,shmem_size=1024,sync=false)
        return ()->JACC.parallel_for(spec,65536,stream_dot!,a,b,partial,n,template)
    end
    code=findfirst(==(op),["copy","mul","add","triad","nstream"]);isnothing(code)&&error("operation")
    spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(spec,n,stream_map!,a,b,c,code)
end
stream_study(factory)
