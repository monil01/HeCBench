# Same source GPU grammar/Thompson construction, with explicit pointer-slot -> Int32 index mapping.
# Shared bytes0:7999 packed in2000 Int32 words;306node/meta Int32words follow (9224B).
@inline ng(s,i)=(@inbounds s[2000+i])
@inline ns!(s,i,v)=(@inbounds s[2000+i]=Int32(v))
@inline function post_put!(s,i,b)
    word=(i-1)÷4+1;shift=((i-1)%4)*8
    @inbounds old=shift==0 ? UInt32(0) : reinterpret(UInt32,s[word])
    @inbounds s[word]=reinterpret(Int32,(old&~(UInt32(255)<<shift))|(UInt32(b)<<shift))
    nothing
end
@inline post_get(s,i)=(@inbounds UInt8((reinterpret(UInt32,s[(i-1)÷4+1])>>(((i-1)%4)*8))&UInt32(255)))
@inline function parse_postfix!(s,re)
    alts=MVector{100,Int32}(undef);atoms=MVector{100,Int32}(undef)
    level=0;nalt=0;natom=0;dst=1;i=17
    @inbounds while re[i]!=0
        c=re[i]
        if c==5
            if natom>1;natom-=1;post_put!(s,dst,27);dst+=1;end
            level+=1;alts[level]=nalt;atoms[level]=natom;nalt=0;natom=0
        elseif c==4
            while natom>1;natom-=1;post_put!(s,dst,27);dst+=1;end
            natom-=1;nalt+=1
        elseif c==6
            while natom>1;natom-=1;post_put!(s,dst,27);dst+=1;end
            natom-=1
            while nalt>0;post_put!(s,dst,4);dst+=1;nalt-=1;end
            nalt=Int(alts[level]);natom=Int(atoms[level])+1;level-=1
        elseif c==1 || c==2 || c==3
            post_put!(s,dst,c);dst+=1
        else
            if natom>1;natom-=1;post_put!(s,dst,27);dst+=1;end
            post_put!(s,dst,c);dst+=1;natom+=1
        end
        i+=1
    end
    while natom>1;natom-=1;post_put!(s,dst,27);dst+=1;end
    while nalt>0;post_put!(s,dst,4);dst+=1;nalt-=1;end
    post_put!(s,dst,0);ns!(s,306,dst-1)
    nothing
end
@inline slot_index(slot)=isodd(slot) ? 101+(slot+1)÷2 : 202+slot÷2
@inline function patch_slots!(s,head,target)
    while head!=0
        index=slot_index(head);next=ng(s,index);ns!(s,index,target);head=next
    end
    nothing
end
@inline function append_slots!(s,left,right)
    old=left;next=ng(s,slot_index(left))
    while next!=0;left=next;next=ng(s,slot_index(left));end
    ns!(s,slot_index(left),right)
    old
end
@inline function new_state!(s,id,c,a,b)
    ns!(s,id,c);ns!(s,101+id,a);ns!(s,202+id,b)
    nothing
end
@inline function build_nfa!(s,match)
    starts=MVector{1000,Int32}(undef);heads=MVector{1000,Int32}(undef)
    count=0;top=0;i=1
    @inbounds while post_get(s,i)!=0
        c=post_get(s,i)
        if c==27
            s2=starts[top];h2=heads[top];top-=1;s1=starts[top];h1=heads[top];top-=1
            patch_slots!(s,h1,s2);top+=1;starts[top]=s1;heads[top]=h2
        elseif c==4
            s2=starts[top];h2=heads[top];top-=1;s1=starts[top];h1=heads[top];top-=1
            count+=1;new_state!(s,count,257,s1,s2);h=append_slots!(s,h1,h2)
            top+=1;starts[top]=count;heads[top]=h
        elseif c==2
            a=starts[top];h=heads[top];top-=1;count+=1;new_state!(s,count,257,a,0)
            h=append_slots!(s,h,2count);top+=1;starts[top]=count;heads[top]=h
        elseif c==3 || c==1
            a=starts[top];h=heads[top];top-=1;count+=1;new_state!(s,count,257,a,0)
            patch_slots!(s,h,count);top+=1;starts[top]=(c==3 ? count : a);heads[top]=2count
        else
            count+=1;new_state!(s,count,c==21 ? 258 : Int(c),0,0)
            top+=1;starts[top]=count;heads[top]=2count-1
        end
        i+=1
    end
    start=starts[top];head=heads[top];patch_slots!(s,head,101)
    ns!(s,101,match[17]);ns!(s,304,count);ns!(s,305,start)
    nothing
end
@inline function epsilon_add!(dest,n,start,s,pending)
    top=1;@inbounds pending[top]=start
    @inbounds while top>0
        state=pending[top];top-=1
        if state!=0
            if ng(s,state)==257
                top+=1;pending[top]=ng(s,101+state);top+=1;pending[top]=ng(s,202+state)
            else
                n+=1;dest[n]=state
            end
        end
    end
    n
end
@inline function match_line(s,text,base)
    one=MVector{100,Int32}(undef);two=MVector{100,Int32}(undef);pending=MVector{100,Int32}(undef)
    current=one;next=two;count=epsilon_add!(current,0,ng(s,305),s,pending);i=base
    @inbounds while text[i]!=0
        char=Int(text[i]);nnext=0
        for j=1:count
            state=current[j];c=ng(s,state)
            if c==char || c==258;nnext=epsilon_add!(next,nnext,ng(s,101+state),s,pending);end
        end
        current,next=next,current;count=nnext;i+=1
    end
    @inbounds for j=1:count
        if ng(s,current[j])==256;return UInt8(1);end
    end
    UInt8(0)
end
@inline function match_thread!(gid,s,text,offset,result,n)
    i=gid
    @inbounds while i<n
        result[17+i]=match_line(s,text,17+Int(offset[17+i]));i+=512*160
    end
    nothing
end
@inline function export_graph!(s,out,block)
    base=17+block*404;len=Int(ng(s,306));count=Int(ng(s,304))
    @inbounds begin
        out[base]=len;out[base+1]=count;out[base+2]=ng(s,305)
        for i=0:len;out[base+3+i]=Int32(post_get(s,i+1));end
        for i=1:count
            out[base+103+(i-1)*3]=ng(s,i)
            out[base+104+(i-1)*3]=ng(s,101+i)
            out[base+105+(i-1)*3]=ng(s,202+i)
        end
    end
    nothing
end
