# CPU-only semantic replay of the actual Julia device helpers; no GPU is visible.
# This supplements the submitted GPU evaluation and is not a timed trial/JIT claim.
include(joinpath(@__DIR__,"thermodynamic_device.jl"))
include(joinpath(@__DIR__,"stage_device.jl"))
const ROOT=@__DIR__
function readwords(name,T)
    open(joinpath(ROOT,name),"r")do f
        @assert filesize(f)%sizeof(T)==0
        x=Vector{T}(undef,filesize(f)÷sizeof(T));read!(f,x);@assert eof(f);x
    end
end
function main()
    par=readwords("params.bin",UInt32);seq=readwords("sequences.bin",Int32);gold=readwords("serial_source_crosscheck.bin",Int32)
    offset=0;k=2
    function check(c,offset)
        for key in (:v,:w,:wm,:wca,:w5,:w3)
            values=getproperty(c,key)
            for got in values
                offset+=1
                if got!=gold[offset];error("CPU_JULIA_CANDIDATE_MISMATCH offset=$offset got=$got source=$(gold[offset])");end
            end
        end
        offset
    end
    for case in 1:seq[1]
        n=seq[k];k+=1;s=seq[k:k+Int(n)-1];k+=Int(n)
        c=(params=par,seq=s,v=fill(RNA_INF,Int(n*n)),w=fill(RNA_INF,Int(n*n)),wm=fill(RNA_INF,Int(n*n)),wca=fill(RNA_INF,Int(n*n)),w5=zeros(Int32,Int(n)+1),w3=zeros(Int32,Int(n)+1))
        offset=check(c,offset)
        for d in Int32(0):n-Int32(2)
            for stage in 1:9
                tag=Val(stage)
                if stage in (1,3,4,6)
                    for cell in Int32(0):n-Int32(1);rna_scalar_stage(tag,c,n,d,cell);end
                else
                    cells=stage==9 ? (Int32(0):Int32(0)) : (Int32(0):n-Int32(1))
                    for cell in cells
                        a=RNA_INF;b=RNA_INF;active=false
                        for lane in Int32(0):Int32(255)
                            tx=stage==2 ? rem(lane,Int32(16)) : lane
                            ty=stage==2 ? div(lane,Int32(16)) : Int32(0)
                            x,y,valid=rna_partial(tag,c,n,d,cell,tx,ty,lane)
                            a=min(a,x);b=min(b,y);active|=valid
                        end
                        rna_finish(tag,c,n,d,cell,a,b,active)
                    end
                end
            end
            offset=check(c,offset)
        end
        println("CPU_SEMANTIC_CASE ",case," n",n," exact_values_through",offset)
    end
    @assert offset==length(gold)&&k==length(seq)+1
    println("CPU_SEMANTIC_REPLAY_PASS fullvalues",offset," noGPUvisible/noGPUpipelineverifiedbythisauxiliarycheck")
end
main()
