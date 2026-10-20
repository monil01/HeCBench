# Source-ordered Float64 operations. Backend entrypoints supply logical indices.
@inline function mini_vector!(i,n,x,y,w,alpha,beta,::Val{O}) where O
    if i<=n
        @inbounds begin
            if O==0 || O==6
                w[i]=muladd(beta,y[i],x[i])
            elseif O==1 || O==7
                w[i]=muladd(alpha,x[i],beta*y[i])
            elseif O==2
                w[i]=x[i]
            elseif O==3
                w[i]=alpha*x[i]
            elseif O==4
                w[i]=y[i]+x[i]
            elseif O==5
                w[i]=muladd(alpha,x[i],y[i])
            end
        end
    end
    nothing
end
@inline function mini_matvec!(i,n,rp,cols,a,x,w)
    if i<=n
        @inbounds begin
            start=rp[i]+Int32(1); stop=rp[i+Int32(1)]; s=0.0
            # Native source #pragma unroll 27; every configured row has <=27 entries.
            Base.Cartesian.@nexprs 27 k -> begin
                j=start+Int32(k-1)
                if j<=stop
                    s=muladd(a[j],x[cols[j]+Int32(1)],s)
                end
            end
            w[i]=s
        end
    end
    nothing
end
