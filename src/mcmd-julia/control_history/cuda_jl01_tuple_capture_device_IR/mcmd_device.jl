# Complete selected source force path: Lennard-Jones + real/reciprocal Ewald.
# Source pform2 polarization and other CLI modes are outside this named contract.
@inline mcmd_at(a,i,k)=@inbounds a[(i-Int32(1))*Int32(14)+k]
@inline mcmd_molid(a,i)=Int32(reinterpret(UInt64,mcmd_at(a,i,Int32(10)))&UInt64(0xffffffff))
@inline mcmd_frozen(a,i)=(reinterpret(UInt64,mcmd_at(a,i,Int32(10)))>>32)!=0
@inline mcmd_rint(x::Float64)=ccall("extern __nv_rint",llvmcall,Float64,(Float64,),x)
@inline mcmd_erfc(x::Float64)=ccall("extern __nv_erfc",llvmcall,Float64,(Float64,),x)
@inline function mcmd_add!(a,i,n,x)
    k=(i-Int32(1))*Int32(14)+Int32(6)+n
    CUDA.@atomic a[k]+=x
    nothing
end
@inline function mcmd_displacement(a,i,j,b,rb)
    d=ntuple(c->mcmd_at(a,i,Int32(c))-mcmd_at(a,j,Int32(c)),Val(3))
    img=ntuple(Val(3)) do c
        t=0.0
        for q in Int32(1):Int32(3);t+=(@inbounds rb[(Int32(c)-Int32(1))*Int32(3)+q])*d[q];end
        mcmd_rint(t)
    end
    di=ntuple(Val(3)) do c
        t=0.0
        for q in Int32(1):Int32(3);t+=(@inbounds b[(Int32(c)-Int32(1))*Int32(3)+q])*img[q];end
        d[c]-t
    end
    ri2=0.0;r2=0.0
    for c in Int32(1):Int32(3);ri2+=di[c]*di[c];r2+=d[c]*d[c];end
    ri=sqrt(ri2)
    isnan(ri) ? (d,r2,sqrt(r2)) : (di,ri2,ri)
end
function mcmd_force_workitem!(i::Int32,a,N::Int32,cutoff,b,rb,alpha,km::Int32,ks::Int32)
    i>N&&return nothing
    mi=mcmd_molid(a,i);fi=mcmd_frozen(a,i);af=(0.0,0.0,0.0)
    # Source LJ pair loop and Lorentz-Berthelot mixing; same register accumulation.
    for j in i+Int32(1):N
        (mi==mcmd_molid(a,j)||(fi&&mcmd_frozen(a,j)))&&continue
        sig=mcmd_at(a,i,Int32(5));sj=mcmd_at(a,j,Int32(5));sig!=sj&&(sig=0.5*(sig+sj))
        eps=mcmd_at(a,i,Int32(4));ej=mcmd_at(a,j,Int32(4));eps!=ej&&(eps=sqrt(eps*ej))
        (sig==0.0||eps==0.0)&&continue
        d,rsq,r=mcmd_displacement(a,i,j,b,rb)
        if r<=cutoff
            r6=rsq*rsq*rsq;s6=sig*sig;s6*=s6*s6
            holder=ntuple(n->24.0*d[n]*eps*(2*(s6*s6)/(r6*r6*rsq)-s6/(r6*rsq)),Val(3))
            for n in Int32(1):Int32(3);mcmd_add!(a,j,n,-holder[n]);end
            af=ntuple(n->af[n]+holder[n],Val(3))
        end
    end
    for n in Int32(1):Int32(3);mcmd_add!(a,i,n,af[n]);end
    # Source electrostatic component including its original volume expression.
    af=(0.0,0.0,0.0);sqrtPI=sqrt(Float64(pi));fourPI=4.0*Float64(pi)
    @inbounds invV=b[1]*(b[5]*b[9]-b[8]*b[6]);@inbounds invV+=b[4]*(b[8]*b[3]-b[2]*b[9]);@inbounds invV+=b[7]*(b[2]*b[6]-b[6]*b[3]);invV=1.0/invV
    qi=mcmd_at(a,i,Int32(6))
    for j in Int32(1):N
        (i==j||(fi&&mcmd_frozen(a,j)))&&continue
        qj=mcmd_at(a,j,Int32(6));(qi==0.0||qj==0.0)&&continue
        d,rsq,r=mcmd_displacement(a,i,j,b,rb);qq=qi*qj
        if r<=cutoff&&mi<mcmd_molid(a,j)
            u=ntuple(n->d[n]/r,Val(3));holder=ntuple(n->-((-2.0*qq*alpha*exp(-alpha*alpha*rsq))/(sqrtPI*r)-(qq*mcmd_erfc(alpha*r)/rsq))*u[n],Val(3))
            af=ntuple(n->af[n]+holder[n],Val(3))
            for n in Int32(1):Int32(3);mcmd_add!(a,j,n,-holder[n]);end
        end
        if ks!=0&&mi<mcmd_molid(a,j)
            for n in Int32(1):Int32(3)
                acc=af[n]
                for l0 in Int32(0):km
                    for l1 in (l0==0 ? Int32(0) : -km):km
                        for l2 in (l0==0&&l1==0 ? Int32(1) : -km):km
                            l0*l0+l1*l1+l2*l2>km*km&&continue
                            l=(l0,l1,l2)
                            k=ntuple(Val(3)) do p
                                s=0.0
                                for q in Int32(1):Int32(3);s+=2.0*Float64(pi)*(@inbounds rb[Int32(3)*(q-Int32(1))+Int32(p)])*l[q];end
                                s
                            end
                            ksq=k[1]*k[1]+k[2]*k[2]+k[3]*k[3]
                            h=qq*invV*fourPI*k[n]*exp(-ksq/(4*alpha*alpha))*sin(k[1]*d[1]+k[2]*d[2]+k[3]*d[3])/ksq*2
                            acc+=h;mcmd_add!(a,j,n,-h)
                        end
                    end
                end
                af=ntuple(c->c==n ? acc : af[c],Val(3))
            end
        end
    end
    for n in Int32(1):Int32(3);mcmd_add!(a,i,n,af[n]);end
    nothing
end
