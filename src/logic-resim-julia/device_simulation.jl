# Full source simulateParallel mathematical/control translation, shared across frontends.
# Original MIT notices retained in LICENSE.source; exact C source in source_kernel.cuh.
using StaticArrays
struct LogicEvent
    t::UInt64
    v::Int8
end
@inline event_at(a,p::UInt64)=@inbounds a[Int(p)+1]
@inline event_put!(a,p::UInt64,e::LogicEvent)=(@inbounds a[Int(p)+1]=e)
@inline function simulate_gate!(gid::Int32,arena,func,input,delaytable,output,capacity,sizes,overflow,truth,n::Int32,dumpOff::UInt64)
    gid>=n && return nothing
    j=gid+Int32(1)
    @inbounds begin
        maxSize=capacity[j]
        base=output[j]
        functionID=func[j]
        inLen=Int32(truth[Int(functionID)*1025+1025])
        port=MVector{6,UInt64}(0,0,0,0,0,0)
        nextTime=MVector{6,UInt64}(0,0,0,0,0,0)
        currentTime=typemax(UInt64)
        initialized=true
        delay=typemax(UInt32)
        previousCode=Int16(0);currentCode=Int16(0)
        previousOutput=Int8(0);currentOutput=Int8(0)
        overflow[j]=false
        i=Int32(1)
        while i<=inLen
            port[i]=input[j,i]
            e=event_at(arena,port[i])
            previousCode=(previousCode<<2)|Int16(e.v)
            nextTime[i]=event_at(arena,port[i]+UInt64(1)).t
            initialized=e.t>0 ? false : initialized
            currentTime=min(currentTime,nextTime[i])
            i+=Int32(1)
        end
        currentSize=UInt64(0)
        if initialized
            currentSize=UInt64(1)
            event_put!(arena,base,LogicEvent(UInt64(0),Int8(2)))
            previousOutput=Int8(2)
        else
            currentSize=sizes[j]
            previousOutput=event_at(arena,base+currentSize-UInt64(1)).v
        end
        if currentTime==typemax(UInt64)||currentTime>dumpOff
            if event_at(arena,base+currentSize-UInt64(1)).t!=typemax(UInt64)
                event_put!(arena,base+currentSize,LogicEvent(typemax(UInt64),Int8(3)))
                sizes[j]=currentSize+UInt64(1)
            end
            return nothing
        end
        while currentTime!=typemax(UInt64)&&currentTime<=dumpOff
            i=Int32(1)
            while i<=inLen
                if nextTime[i]==currentTime
                    port[i]+=UInt64(1)
                    nextTime[i]=event_at(arena,port[i]+UInt64(1)).t
                end
                currentCode=(currentCode<<2)|Int16(event_at(arena,port[i]).v)
                i+=Int32(1)
            end
            code=Int32(currentCode)
            word=truth[Int(functionID)*1025+(Int(code)>>2)+1]
            currentOutput=Int8((word>>((code&Int32(3))*Int32(2)))&UInt8(3))
            if currentOutput!=previousOutput
                transition=(Int32(previousOutput)<<2)|Int32(currentOutput)
                edge=transition==1 ? Int32(0) : transition==4 ? Int32(1) : transition==2 ? Int32(2) : transition==9 ? Int32(3) : transition==6 ? Int32(4) : Int32(5)
                i=Int32(1)
                while i<=inLen
                    shift=(inLen-i)*Int32(2)
                    to=Int8((currentCode>>shift)&Int16(3))
                    from=Int8((previousCode>>shift)&Int16(3))
                    if to!=from
                        direction=(from==Int8(1)||to==Int8(0)) ? Int32(6) : Int32(0)
                        d=delaytable[j,Int32(12)*(i-Int32(1))+edge+direction+Int32(1)]
                        delay=min(delay,d)
                    end
                    i+=Int32(1)
                end
                currentTime+=UInt64(delay)
                while currentSize>0&&event_at(arena,base+currentSize-UInt64(1)).t>=currentTime
                    currentSize-=UInt64(1)
                end
                if currentSize==0||event_at(arena,base+currentSize-UInt64(1)).v!=currentOutput
                    event_put!(arena,base+currentSize,LogicEvent(currentTime,currentOutput))
                    currentSize+=UInt64(1)
                    if currentSize==maxSize
                        overflow[j]=true
                        sizes[j]=currentSize
                        return nothing
                    end
                end
            end
            previousCode=currentCode;previousOutput=currentOutput
            currentCode=Int16(0);currentOutput=Int8(0)
            currentTime=typemax(UInt64);delay=typemax(UInt32)
            i=Int32(1)
            while i<=inLen
                currentTime=min(currentTime,nextTime[i])
                i+=Int32(1)
            end
        end
        event_put!(arena,base+currentSize,LogicEvent(typemax(UInt64),Int8(3)))
        sizes[j]=currentSize+UInt64(1)
    end
    nothing
end
