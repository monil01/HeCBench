using CUDA,SHA
@inline function ss_match(text,start,pattern,offset,length)
    k=Int32(0)
    while k<length
        @inbounds ss_lower(text[start+k+Int32(1)])==pattern[offset+k+Int32(1)]||return false
        k+=Int32(1)
    end
    true
end
function ss_naive!(text,n,pattern,p,result,counts,width)
    tid=Int32(threadIdx().x)-Int32(1);nt=Int32(blockDim().x);group=Int32(blockIdx().x)-Int32(1)
    localpattern=CUDA.CuDynamicSharedArray(UInt8,p);success=CUDA.CuStaticSharedArray(UInt32,1)
    last=n-p+Int32(1);beginpos=group*width;endpos=min(beginpos+width,last)
    if beginpos<=last
        i=tid
        while i<p
            @inbounds localpattern[i+Int32(1)]=ss_lower(pattern[i+Int32(1)]);i+=nt
        end
        if tid==0;@inbounds success[1]=UInt32(0);end
        CUDA.sync_threads();position=beginpos+tid
        while position<endpos
            if ss_match(text,position,localpattern,Int32(0),p)
                slot=CUDA.atomic_add!(pointer(success,1),UInt32(1))
                @inbounds result[beginpos+Int32(slot)+Int32(1)]=UInt32(position)
            end
            position+=nt
        end
        CUDA.sync_threads()
        if tid==0;@inbounds counts[group+Int32(1)]=success[1];end
    end
    nothing
end
function ss_load_balance!(text,n,pattern,p,result,counts,width)
    tid=Int32(threadIdx().x)-Int32(1);nt=Int32(blockDim().x);group=Int32(blockIdx().x)-Int32(1)
    localpattern=CUDA.CuDynamicSharedArray(UInt8,p)
    stack1=CUDA.CuStaticSharedArray(UInt32,512);stack2=CUDA.CuStaticSharedArray(UInt32,512)
    counter=CUDA.CuStaticSharedArray(UInt32,3)
    @inbounds if tid==0;counter[1]=UInt32(0);counter[2]=UInt32(0);counter[3]=UInt32(0);end
    last=n-p+Int32(1);beginpos=group*width;endpos=min(beginpos+width,last);searchlen=endpos-beginpos
    if beginpos<=last
        i=tid
        while i<p
            @inbounds localpattern[i+Int32(1)]=ss_lower(pattern[i+Int32(1)]);i+=nt
        end
        CUDA.sync_threads()
        @inbounds first=localpattern[1];@inbounds second=localpattern[2]
        position=tid
        while true
            if position<searchlen
                @inbounds initial=(first==ss_lower(text[beginpos+position+Int32(1)]))&&(second==ss_lower(text[beginpos+position+Int32(2)]))
                if initial
                    slot=CUDA.atomic_add!(pointer(counter,1),UInt32(1));@inbounds stack1[Int32(slot)+Int32(1)]=UInt32(position)
                end
            end
            position+=nt
            CUDA.sync_threads();@inbounds size=Int32(counter[1]);CUDA.sync_threads()
            if size<nt&&(position÷nt)*nt<searchlen;continue;end
            if tid<size
                previous=CUDA.atomic_sub!(pointer(counter,1),UInt32(1));@inbounds pos=Int32(stack1[Int32(previous)])
                status=true
                for k in Int32(2):Int32(9)
                    @inbounds status =status&&(localpattern[k+Int32(1)]==ss_lower(text[beginpos+pos+k+Int32(1)]))
                end
                if status
                    slot=CUDA.atomic_add!(pointer(counter,2),UInt32(1));@inbounds stack2[Int32(slot)+Int32(1)]=UInt32(pos)
                end
            end
            CUDA.sync_threads();@inbounds size=Int32(counter[2]);CUDA.sync_threads()
            if size<nt&&(position÷nt)*nt<searchlen;continue;end
            if tid<size
                previous=CUDA.atomic_sub!(pointer(counter,2),UInt32(1));@inbounds pos=Int32(stack2[Int32(previous)])
                if ss_match(text,beginpos+pos+Int32(10),localpattern,Int32(10),p-Int32(10))
                    slot=CUDA.atomic_add!(pointer(counter,3),UInt32(1));@inbounds result[beginpos+Int32(slot)+Int32(1)]=UInt32(beginpos+pos)
                end
            end
            CUDA.sync_threads()
            @inbounds done=(position÷nt)*nt>=searchlen&&counter[1]==UInt32(0)&&counter[2]==UInt32(0)
            if done;break;end
        end
        if tid==0;@inbounds counts[group+Int32(1)]=counter[3];end
    end
    nothing
end
function ss_launch(::Val{0},text,pattern,result,counts,n,p,width,groups)
    @cuda threads=256 blocks=groups shmem=Int(p) stream=SS_STREAM ss_naive!(text,n,pattern,p,result,counts,width)
    SS_COUNTS[1]+=1;nothing
end
function ss_launch(::Val{1},text,pattern,result,counts,n,p,width,groups)
    @cuda threads=256 blocks=groups shmem=Int(p) stream=SS_STREAM ss_load_balance!(text,n,pattern,p,result,counts,width)
    SS_COUNTS[2]+=1;nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
ss_study()
