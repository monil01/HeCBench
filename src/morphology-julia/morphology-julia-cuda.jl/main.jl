using CUDA
using Printf
# Correction: native vHGW uses zero horizontal margins and operation-specific vertical margins.
function morphology_kernel!(dst,src,W::Int32,H::Int32,hs::Int32,vs::Int32,op::Int32,axis::Int32)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
    if i<W*H
        x=i%W;y=i÷W;sel=axis==Int32(1) ? hs : vs;half=sel÷Int32(2)
        position=axis==Int32(1) ? x : y;limit=axis==Int32(1) ? W : H
        if position<half || position>=limit-half
            @inbounds dst[i+Int32(1)]=axis==Int32(1) ? UInt8(0) : (op==Int32(1) ? UInt8(255) : UInt8(0))
        else
            acc=op==Int32(1) ? UInt8(0) : UInt8(255)
            for k in -half:half
                j=axis==Int32(1) ? i+k : i+k*W
                @inbounds value=src[j+Int32(1)]
                acc=op==Int32(1) ? max(acc,value) : min(acc,value)
            end
            @inbounds dst[i+Int32(1)]=acc
        end
    end
    nothing
end
function morphology_cpu_axis!(dst,src,W,H,sel,op,axis)
    half=sel÷2
    for y in 0:H-1,x in 0:W-1
        i=y*W+x+1;position=axis==1 ? x : y;limit=axis==1 ? W : H
        if position<half || position>=limit-half
            dst[i]=axis==1 ? UInt8(0) : (op==1 ? UInt8(255) : UInt8(0))
        else
            acc=op==1 ? UInt8(0) : UInt8(255)
            for k in -half:half
                j=axis==1 ? i+k : i+k*W;value=src[j];acc=op==1 ? max(acc,value) : min(acc,value)
            end
            dst[i]=acc
        end
    end
end
function launch_morphology!(img,tmp,W,H,hs,vs,op)
    CUDA.fill!(tmp,UInt8(0));CUDA.synchronize();t0=time_ns()
    @cuda threads=256 blocks=cld(W*H,256) morphology_kernel!(tmp,img,Int32(W),Int32(H),Int32(hs),Int32(vs),Int32(op),Int32(1))
    @cuda threads=256 blocks=cld(W*H,256) morphology_kernel!(img,tmp,Int32(W),Int32(H),Int32(hs),Int32(vs),Int32(op),Int32(2))
    CUDA.synchronize();time_ns()-t0
end
function main(args=ARGS)
    length(args)==5||error("Usage: main.jl kernel_width kernel_height image_width image_height repeat")
    hs,vs,W,H,repeats=parse.(Int,args);isodd(hs)&&isodd(vs)&&hs>0&&vs>0&&W>=hs&&H>=vs&&repeats>0||error("positive odd windows and valid image/repeats required")
    input=zeros(UInt8,W*H);input[(H÷2-1)*W+(W÷2-1)+1]=255
    expected=copy(input);tmp=similar(expected)
    for _ in 1:repeats,op in (1,2)
        morphology_cpu_axis!(tmp,expected,W,H,hs,op,1);morphology_cpu_axis!(expected,tmp,W,H,vs,op,2)
    end
    a=CuArray(input);b=similar(a);times=zeros(Float64,2)
    for _ in 1:repeats,op in (1,2)
        times[op]+=launch_morphology!(a,b,W,H,hs,vs,op)
    end
    @printf("Average kernel execution time (dilate): %f (s)\n",times[1]*1e-9/repeats)
    @printf("Average kernel execution time (erode): %f (s)\n",times[2]*1e-9/repeats)
    ok=Array(a)==expected;println(ok ? "PASS" : "FAIL");ok ? 0 : 1
end
exit(main())
