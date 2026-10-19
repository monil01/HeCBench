# Source-scaled private recurrence; terminal inactive-lookup guard matches qualified native.
# Named contract L33,D50; source first/otherGap ignored and threshold .5 preserved.
@inline function repeat_probs!(pr,seq,base,mat,p)
    fg=MVector{50,Float64}(undef);lookup=MVector{50,Float64}(undef);scales=MVector{2,Float64}(undef)
    b2b=1-p.rp;f2f=1-p.re;f2b=p.re;growth=1/p.decay
    first=growth<1 || growth>1 ? (1-growth)/(1-growth^Float64(p.d)) : 1.0/p.d
    last=p.rp*first;lastinv=1/last;v=last
    for i=1:p.d
        lookup[i]=v;v*=growth;fg[i]=0.0
    end
    bg=1.0
    for k=0:p.l-1
        v0=Int(seq[base+k]);cap=min(k,p.d)
        pad1=cap-1;pad2=p.d-cap;pad3=k-cap;acc=0.0
        for i=0:k-1
            i1=pad1-i+1;i2=pad2+i+1;i3=pad3+i
            v1=Int(seq[base+i3]);acc+=fg[i1]
            fg[i1]=(f2f*fg[i1]+bg*lookup[i2])*mat[17+v0*p.l+v1]
        end
        bg=bg*b2b+acc*f2b
        if k%16==15
            scale=1/bg;scales[k÷16+1]=scale
            for i=1:cap;fg[i]*=scale;end
            bg=1.0
        end
        pr[k+1]=Float32(bg)
    end
    acc=0.0
    for i=1:p.d;acc+=fg[i];fg[i]=f2b;end
    ftot=bg*b2b+acc*f2b;bg=b2b;invtotal=1/ftot
    for k=p.l-1:-1:0
        nonrepeat=Float32(Float64(pr[k+1])*bg*invtotal);pr[k+1]=1f0-nonrepeat
        cap=min(k,p.d)
        if k%16==15
            scale=scales[k÷16+1]
            for i=1:cap;fg[i]*=scale;end
            bg*=scale
        end
        c0=f2b*bg;v0=Int(seq[base+k]);acc=0.0
        for i=0:cap-1
            v1=Int(seq[base+k-(i+1)]);f=fg[i+1]*mat[17+v0*p.l+v1]
            acc+=lookup[cap-i]*f;fg[i+1]=c0+f2f*f
        end
        coeff=(cap==0 || k>p.d) ? 1.0 : lookup[p.d-k+1]*lastinv
        bg=b2b*bg+acc*coeff
    end
    Int32(abs(ftot-bg)>max(ftot,bg)/1e6)
end
@inline function mask_sequence!(seq,mat,table,p,gid)
    pr=MVector{33,Float32}(undef);base=17+gid*33
    repeat_probs!(pr,seq,base,mat,p)
    for i=0:p.l-1
        if pr[i+1]>=.5;seq[base+i]=table[17+Int(seq[base+i])];end
    end
    nothing
end
@inline function observe_sequence!(seq,mat,p,post,errors,gid)
    pr=MVector{33,Float32}(undef);base=17+gid*33
    err=repeat_probs!(pr,seq,base,mat,p);errors[17+gid]=err
    for i=0:p.l-1;post[base+i]=pr[i+1];end
    nothing
end
