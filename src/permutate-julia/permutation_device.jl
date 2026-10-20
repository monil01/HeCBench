# Authored source-behavior helpers. The original known shuffle/terminal-run defects
# remain intentional. Collision seen[256] is represented by four UInt64 words:
# exact membership, different storage/clearing cost; performance covariate.
@inline perm_get(a,i)=(@inbounds a[i+1])
@inline function perm_shuffle!(S,B,D,n,N,tid,::Val{Binary}) where Binary
    state=UInt64(0)
    for i in 0:n-1
        x=perm_get(D,i);@inbounds S[i*N+tid+1]=x;state+=UInt64(x)
    end
    state=state ⊻ UInt64(tid)
    j=n-1
    while j>0
        state=(UInt64(2806196910506780709)*state+UInt64(1))&UInt64(0x7fffffffffffffff)
        k=Int(UInt32(state&UInt64(0xffffffff)))%j
        a=k*N+tid+1;b=j*N+tid+1
        @inbounds x=S[a];@inbounds S[a]=S[b];@inbounds S[b]=x;j-=1
    end
    if Binary
        for i in 0:div(n,8)-1
            byte=UInt8(0)
            for j in 0:7;byte |= (perm_get(S,(8*i+j)*N+tid)&UInt8(1))<<(7-j);end
            @inbounds B[i*N+tid+1]=byte
        end
    end
    nothing
end
@inline function perm_excursion(a,n,N,tid,mean)
    cumulative=0.0;largest=0.0
    for i in 0:n-1
        cumulative+=Float64(perm_get(a,i*N+tid));v=abs(fma(-Float64(i+1),mean,cumulative));largest=max(largest,v)
    end
    largest
end
@inline function perm_direction_median(a,n,N,tid,median)
    ndirection=1.0;nmedian=1.0;longdirection=0.0;longmedian=0.0
    run1=1;run2=1;increases=0
    f1=perm_get(a,tid)<=perm_get(a,N+tid);fm1=Float64(perm_get(a,tid))>=median
    for i in 1:n-2
        increases+=Int(f1);f2=perm_get(a,i*N+tid)<=perm_get(a,(i+1)*N+tid);fm2=Float64(perm_get(a,i*N+tid))>=median
        if f1==f2;run1+=1;else;ndirection+=1.0;longdirection=max(longdirection,Float64(run1));run1=1;end
        if fm1==fm2;run2+=1;else;nmedian+=1.0;longmedian=max(longmedian,Float64(run2));run2=1;end
        f1=f2;fm1=fm2
    end
    increases+=Int(f1);fm2=Float64(perm_get(a,(n-1)*N+tid))>=median
    if fm1==fm2;run2+=1;else;nmedian+=1.0;longmedian=max(longmedian,Float64(run2));run2=1;end
    ndirection,longdirection,Float64(max(increases,n-increases)),nmedian,longmedian
end
@inline function perm_median(a,n,N,tid,median)
    runs=1;finished=0;current=1;previous=Float64(perm_get(a,tid))>=median
    for i in 1:n-1
        next=Float64(perm_get(a,i*N+tid))>=median
        if previous==next;current+=1;else;runs+=1;finished=max(finished,current);current=1;end
        previous=next
    end
    Float64(runs),Float64(finished)
end
@inline perm_hamming(x::UInt8)=Int(count_ones(x))
@inline function perm_binary_direction(a,n,N,tid)
    runs=1;finished=0;current=1;increases=0;previous=perm_hamming(perm_get(a,tid))<=perm_hamming(perm_get(a,N+tid))
    for i in 1:n-2
        increases+=Int(previous);next=perm_hamming(perm_get(a,i*N+tid))<=perm_hamming(perm_get(a,(i+1)*N+tid))
        if previous==next;current+=1;else;runs+=1;finished=max(finished,current);current=1;end
        previous=next
    end
    increases+=Int(previous);Float64(runs),Float64(finished),Float64(max(increases,n-increases))
end
@inline function perm_collision(a,n,N,tid)
    start=0;collisions=0;sumlength=0.0;longest=0
    while start<n
        seen=(UInt64(0),UInt64(0),UInt64(0),UInt64(0));position=start;found=false
        while position<n
            value=Int(perm_get(a,position*N+tid));word=(value>>6)+1;bit=UInt64(1)<<(value&63)
            if (seen[word]&bit)!=0
                distance=position-start;sumlength+=Float64(distance);longest=max(longest,distance);collisions+=1;start=position+1;found=true;break
            end
            seen=Base.setindex(seen,seen[word]|bit,word);position+=1
        end
        if !found;break;end
    end
    sumlength/Float64(collisions),Float64(longest)
end
@inline function perm_period(a,n,N,tid,lag,::Val{Binary}) where Binary
    equal=0.0;product=0.0
    for i in 0:n-lag-1
        x=perm_get(a,i*N+tid);y=perm_get(a,(i+lag)*N+tid)
        if Binary
            xx=perm_hamming(x);yy=perm_hamming(y);equal+=Float64(xx==yy);product+=Float64(xx*yy)
        else
            equal+=Float64(x==y);product+=Float64(Int(x)*Int(y))
        end
    end
    equal,product
end
@inline function perm_rank!(counts,baseline,k,value)
    reference=@inbounds baseline[k]
    if k==1;value=Float64(Float32(value));reference=Float64(Float32(reference));end
    category=value>reference ? 0 : value==reference ? 1 : 2
    CUDA.atomic_add!(pointer(counts,3*(k-1)+category+1),UInt32(1))
    nothing
end
@inline function perm_stats_generic!(S,B,R,C,n,N,tid,group,mean,median)
    if group==0
        x,y=perm_collision(S,n,N,tid);perm_rank!(C,R,7,x);perm_rank!(C,R,8,y)
    else
        perm_rank!(C,R,1,perm_excursion(S,n,N,tid,mean))
        values=perm_direction_median(S,n,N,tid,median)
        for k in 1:5;perm_rank!(C,R,k+1,values[k]);end
        for k in 1:5
            lag=(1,2,8,16,32)[k];x,y=perm_period(S,n,N,tid,lag,Val(false));perm_rank!(C,R,8+k,x);perm_rank!(C,R,13+k,y)
        end
    end
    nothing
end
@inline function perm_stats_binary!(S,B,R,C,n,N,tid,group,mean,median)
    blen=div(n,8)
    if group==0;perm_rank!(C,R,1,perm_excursion(S,n,N,tid,mean))
    elseif group==1
        x,y=perm_median(S,n,N,tid,median);perm_rank!(C,R,5,x);perm_rank!(C,R,6,y)
        a,b,c=perm_binary_direction(B,blen,N,tid);perm_rank!(C,R,2,a);perm_rank!(C,R,3,b);perm_rank!(C,R,4,c)
    elseif group==2
        x,y=perm_collision(B,blen,N,tid);perm_rank!(C,R,7,x);perm_rank!(C,R,8,y)
    else
        for k in 1:5
            lag=(1,2,8,16,32)[k];x,y=perm_period(B,blen,N,tid,lag,Val(true));perm_rank!(C,R,8+k,x);perm_rank!(C,R,13+k,y)
        end
    end
    nothing
end
