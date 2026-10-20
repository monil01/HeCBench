# Complete source-selected pform1 LJ + real/reciprocal Ewald force path.
# Other CLI potential modes, notably polarization, are outside this contract.
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
@inline function mcmd_row(a,p,x,y,z)
    s=0.0
    for q in Int32(1):Int32(3)
        v=q==1 ? x : q==2 ? y : z
        s+=(@inbounds a[(p-Int32(1))*Int32(3)+q])*v
    end
    s
end
@inline function mcmd_displacement(a,i,j,b,rb)
    x=mcmd_at(a,i,Int32(1))-mcmd_at(a,j,Int32(1));y=mcmd_at(a,i,Int32(2))-mcmd_at(a,j,Int32(2));z=mcmd_at(a,i,Int32(3))-mcmd_at(a,j,Int32(3))
    ix=mcmd_rint(mcmd_row(rb,Int32(1),x,y,z));iy=mcmd_rint(mcmd_row(rb,Int32(2),x,y,z));iz=mcmd_rint(mcmd_row(rb,Int32(3),x,y,z))
    dx=x-mcmd_row(b,Int32(1),ix,iy,iz);dy=y-mcmd_row(b,Int32(2),ix,iy,iz);dz=z-mcmd_row(b,Int32(3),ix,iy,iz)
    r2=0.0;r2+=x*x;r2+=y*y;r2+=z*z;ri2=0.0;ri2+=dx*dx;ri2+=dy*dy;ri2+=dz*dz;ri=sqrt(ri2)
    isnan(ri) ? (x,y,z,r2,sqrt(r2)) : (dx,dy,dz,ri2,ri)
end
@inline function mcmd_kcomponent(rb,p,l0,l1,l2)
    s=0.0
    for q in Int32(1):Int32(3)
        l=q==1 ? l0 : q==2 ? l1 : l2
        s+=2.0*Float64(pi)*(@inbounds rb[Int32(3)*(q-Int32(1))+p])*l
    end
    s
end
function mcmd_force_workitem!(i::Int32,a,N::Int32,cutoff,b,rb,alpha,km::Int32,ks::Int32)
    i>N&&return nothing
    mi=mcmd_molid(a,i);fi=mcmd_frozen(a,i);af1=0.0;af2=0.0;af3=0.0
    for j in i+Int32(1):N
        (mi==mcmd_molid(a,j)||(fi&&mcmd_frozen(a,j)))&&continue
        sig=mcmd_at(a,i,Int32(5));sj=mcmd_at(a,j,Int32(5));sig!=sj&&(sig=0.5*(sig+sj))
        eps=mcmd_at(a,i,Int32(4));ej=mcmd_at(a,j,Int32(4));eps!=ej&&(eps=sqrt(eps*ej))
        (sig==0.0||eps==0.0)&&continue
        dx,dy,dz,rsq,r=mcmd_displacement(a,i,j,b,rb)
        if r<=cutoff
            r6=rsq*rsq*rsq;s6=sig*sig;s6*=s6*s6
            h1=24.0*dx*eps*(2*(s6*s6)/(r6*r6*rsq)-s6/(r6*rsq));h2=24.0*dy*eps*(2*(s6*s6)/(r6*r6*rsq)-s6/(r6*rsq));h3=24.0*dz*eps*(2*(s6*s6)/(r6*r6*rsq)-s6/(r6*rsq))
            mcmd_add!(a,j,Int32(1),-h1);mcmd_add!(a,j,Int32(2),-h2);mcmd_add!(a,j,Int32(3),-h3);af1+=h1;af2+=h2;af3+=h3
        end
    end
    mcmd_add!(a,i,Int32(1),af1);mcmd_add!(a,i,Int32(2),af2);mcmd_add!(a,i,Int32(3),af3)
    af1=0.0;af2=0.0;af3=0.0;sqrtPI=sqrt(Float64(pi));fourPI=4.0*Float64(pi)
    # Preserve the original volume expression; only its orthogonal source box
    # is in this contract, not a claim for arbitrary triclinic cells.
    @inbounds invV=b[1]*(b[5]*b[9]-b[8]*b[6]);@inbounds invV+=b[4]*(b[8]*b[3]-b[2]*b[9]);@inbounds invV+=b[7]*(b[2]*b[6]-b[6]*b[3]);invV=1.0/invV
    qi=mcmd_at(a,i,Int32(6))
    for j in Int32(1):N
        (i==j||(fi&&mcmd_frozen(a,j)))&&continue
        qj=mcmd_at(a,j,Int32(6));(qi==0.0||qj==0.0)&&continue
        dx,dy,dz,rsq,r=mcmd_displacement(a,i,j,b,rb);qq=qi*qj
        if r<=cutoff&&mi<mcmd_molid(a,j)
            h=-((-2.0*qq*alpha*exp(-alpha*alpha*rsq))/(sqrtPI*r)-(qq*mcmd_erfc(alpha*r)/rsq))
            h1=h*(dx/r);h2=h*(dy/r);h3=h*(dz/r);af1+=h1;af2+=h2;af3+=h3
            mcmd_add!(a,j,Int32(1),-h1);mcmd_add!(a,j,Int32(2),-h2);mcmd_add!(a,j,Int32(3),-h3)
        end
        if ks!=0&&mi<mcmd_molid(a,j)
            for n in Int32(1):Int32(3)
                acc=n==1 ? af1 : n==2 ? af2 : af3
                for l0 in Int32(0):km
                    for l1 in (l0==0 ? Int32(0) : -km):km
                        for l2 in (l0==0&&l1==0 ? Int32(1) : -km):km
                            l0*l0+l1*l1+l2*l2>km*km&&continue
                            k1=mcmd_kcomponent(rb,Int32(1),l0,l1,l2);k2=mcmd_kcomponent(rb,Int32(2),l0,l1,l2);k3=mcmd_kcomponent(rb,Int32(3),l0,l1,l2)
                            ksq=k1*k1+k2*k2+k3*k3;kn=n==1 ? k1 : n==2 ? k2 : k3
                            h=qq*invV*fourPI*kn*exp(-ksq/(4*alpha*alpha))*sin(k1*dx+k2*dy+k3*dz)/ksq*2
                            acc+=h;mcmd_add!(a,j,n,-h)
                        end
                    end
                end
                if n==1;af1=acc;elseif n==2;af2=acc;else;af3=acc;end
            end
        end
    end
    mcmd_add!(a,i,Int32(1),af1);mcmd_add!(a,i,Int32(2),af2);mcmd_add!(a,i,Int32(3),af3)
    nothing
end
