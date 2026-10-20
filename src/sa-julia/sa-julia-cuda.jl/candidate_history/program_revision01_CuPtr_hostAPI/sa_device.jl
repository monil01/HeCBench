# Source-faithful DC3 custom device bodies, zero-based source indices explicit.
# Attribution: original sa-cuda/kernels.h, Gangliao (2014); no license inferred.
@inline sa_leq(a,b,c,d) = a<c || (a==c && b<=d)
@inline sa_leq2(a,b,c,d,e,f) = a<d || (a==d && sa_leq(b,c,e,f))
@inline sa_get(a,i) = @inbounds a[i+Int32(1)]
@inline function sa_init!(idx,a,n)
    if idx<n; @inbounds a[idx+Int32(1)]=idx+idx÷Int32(2)+Int32(1);end
    nothing
end
@inline function sa_key!(idx,keys,pos,s,n,offset)
    if idx<n; @inbounds keys[idx+Int32(1)]=sa_get(s,sa_get(pos,idx)+offset);end
    nothing
end
@inline function sa_scanflags!(idx,s,pos,scan,n)
    if idx<n
        a=sa_get(pos,idx);b=sa_get(pos,idx+Int32(1))
        same=sa_get(s,a)==sa_get(s,b)&&sa_get(s,a+Int32(1))==sa_get(s,b+Int32(1))&&sa_get(s,a+Int32(2))==sa_get(s,b+Int32(2))
        @inbounds scan[idx+Int32(1)]=same ? Int32(0) : Int32(1)
    end
    nothing
end
@inline function sa_setrank!(idx,s12,pos,scan,n,n0)
    if idx<n
        a=sa_get(pos,idx);j=a÷Int32(3)+(a%Int32(3)-Int32(1))*n0
        @inbounds s12[j+Int32(1)]=sa_get(scan,idx)+Int32(1)
    end
    nothing
end
@inline function sa_storerank!(idx,s12,pos,n)
    if idx<n;@inbounds s12[sa_get(pos,idx)+Int32(1)]=idx+Int32(1);end
    nothing
end
@inline function sa_unique!(idx,s12,pos,n)
    if idx<n;@inbounds pos[sa_get(s12,idx)]=idx;end
    nothing
end
@inline function sa_mod0flags!(idx,pos,scan,n0,n)
    if idx<n;@inbounds scan[idx+Int32(1)]=sa_get(pos,idx)<n0 ? Int32(1) : Int32(0);end
    nothing
end
@inline function sa_setmod0!(idx,s0,pos,scan,n0,n)
    if idx<n && sa_get(pos,idx)<n0
        @inbounds s0[sa_get(scan,idx)+Int32(1)]=Int32(3)*sa_get(pos,idx)
    end
    nothing
end
@inline function sa_merge!(idx,sa0,sa12,sa,s,s12,n0,n02,n)
    if idx<n0+n02
        flag=Int32(n!=n0+n02)
        if !(flag==1 && idx==n0)
            if idx<n0
                i=sa_get(sa0,idx);left=n0;right=n0+n02
                while left<right
                    mid=(left+right)÷Int32(2);r=sa_get(sa12,mid-n0)
                    if r<n0
                        j=r*Int32(3)+Int32(1)
                        cmp=sa_leq(sa_get(s,j),sa_get(s12,(j+Int32(1))÷Int32(3)+((j+Int32(1))%Int32(3)-Int32(1))*n0),sa_get(s,i),sa_get(s12,i÷Int32(3)))
                    else
                        j=(r-n0)*Int32(3)+Int32(2)
                        cmp=sa_leq2(sa_get(s,j),sa_get(s,j+Int32(1)),sa_get(s12,(j+Int32(2))÷Int32(3)+((j+Int32(2))%Int32(3)-Int32(1))*n0),sa_get(s,i),sa_get(s,i+Int32(1)),sa_get(s12,i÷Int32(3)+n0))
                    end
                    if cmp;left=mid+Int32(1);else;right=mid;end
                end
                @inbounds sa[idx+left-n0-flag+Int32(1)]=i
            else
                r=sa_get(sa12,idx-n0);left=Int32(0);right=n0
                if r<n0
                    i=r*Int32(3)+Int32(1)
                    while left<right
                        mid=(left+right)÷Int32(2);a=sa_get(sa0,mid)
                        cmp=sa_leq(sa_get(s,a),sa_get(s12,a÷Int32(3)),sa_get(s,i),sa_get(s12,(i+Int32(1))÷Int32(3)+((i+Int32(1))%Int32(3)-Int32(1))*n0))
                        if cmp;left=mid+Int32(1);else;right=mid;end
                    end
                else
                    i=(r-n0)*Int32(3)+Int32(2)
                    while left<right
                        mid=(left+right)÷Int32(2);a=sa_get(sa0,mid)
                        cmp=sa_leq2(sa_get(s,a),sa_get(s,a+Int32(1)),sa_get(s12,a÷Int32(3)+n0),sa_get(s,i),sa_get(s,i+Int32(1)),sa_get(s12,(i+Int32(2))÷Int32(3)+((i+Int32(2))%Int32(3)-Int32(1))*n0))
                        if cmp;left=mid+Int32(1);else;right=mid;end
                    end
                end
                @inbounds sa[idx-n0+left-flag+Int32(1)]=i
            end
        end
    end
    nothing
end
