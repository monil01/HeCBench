# GROMACS source and LGPL notices retained in reference_history/COPYRIGHT.
# Mathematical translation of the source 8x8 cluster kernel. CUDA backend
# primitives retain RN-rsqrt, full-mask warp shuffles and Float32 atomics.
# The tested LLVM vector-load/RN-rsqrt idiom is reused from halo-finder's port;
# the force, mask, shift and reduction algorithm below is specific to nbnxm.
@inline nb_rsqrt_rn(x::Float32)=ccall("extern __nv_frsqrt_rn",llvmcall,Float32,(Float32,),x)
@inline function nb_load4(a,atom)
    p=reinterpret(Core.LLVMPtr{NTuple{4,VecElement{Float32}},CUDA.AS.Global},pointer(a,Int32(4)*atom+Int32(1)))
    v=CUDA.unsafe_cached_load(p,1,Val(16))
    (v[1].value,v[2].value,v[3].value,v[4].value)
end
@inline function nb_pme(z2)
    z4=z2*z2
    fd0=muladd(0.0011193462567257629232f0,z4,0.11583842382862377919f0)
    fd1=muladd(0.014866955030185295499f0,z4,0.50736591960530292870f0)
    fd0=muladd(fd0,z4,1f0);fd0=muladd(fd1,z2,fd0);fd0=1f0/fd0
    fn0=muladd(-1.7357322914161492954f-8,z4,-0.000053401640219807709149f0)
    fn1=muladd(1.4703624142580877519f-6,z4,0.0010054721316683106153f0)
    fn0=muladd(fn0,z4,-0.019278317264888380590f0)
    fn1=muladd(fn1,z4,0.069670166153766424023f0)
    fn0=muladd(fn0,z4,-0.75225204789749321333f0)
    fn0=muladd(fn1,z2,fn0)
    fn0*fd0
end
@inline nb_atomic!(a,i,v)=(CUDA.atomic_add!(pointer(a,i),v);nothing)
@inline function nb_preload!(ix,jy,sci,shift,xq,shiftvec,types,smx,smt,eps)
    localAtom=jy*Int32(8)+ix
    atom=(sci*Int32(8)+jy)*Int32(8)+ix
    q=nb_load4(xq,atom)
    @inbounds begin
        smx[Int32(4)*localAtom+Int32(1)]=q[1]+shiftvec[Int32(3)*shift+Int32(1)]
        smx[Int32(4)*localAtom+Int32(2)]=q[2]+shiftvec[Int32(3)*shift+Int32(2)]
        smx[Int32(4)*localAtom+Int32(3)]=q[3]+shiftvec[Int32(3)*shift+Int32(3)]
        smx[Int32(4)*localAtom+Int32(4)]=q[4]*eps
        smt[localAtom+Int32(1)]=types[atom+Int32(1)]
    end
    nothing
end
@inline function nb_reduce_j!(fx,fy,fz,ix,atom,f)
    mask=UInt32(0xffffffff)
    fx+=CUDA.shfl_down_sync(mask,fx,1)
    fy+=CUDA.shfl_up_sync(mask,fy,1)
    fz+=CUDA.shfl_down_sync(mask,fz,1)
    if (ix&Int32(1))!=0;fx=fy;end
    fx+=CUDA.shfl_down_sync(mask,fx,2)
    fz+=CUDA.shfl_up_sync(mask,fz,2)
    if (ix&Int32(2))!=0;fx=fz;end
    fx+=CUDA.shfl_down_sync(mask,fx,4)
    if ix<Int32(3);nb_atomic!(f,Int32(3)*atom+ix+Int32(1),fx);end
    nothing
end
@inline function nb_pairs!(ix,jy,sci,shift,jstart,jend,xq,f,cj,ex,types,coeff,smx,smt,numtypes,rc2,beta)
    fxI=(0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)
    fyI=(0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)
    fzI=(0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)
    tid=jy*Int32(8)+ix;warp=tid÷Int32(32)
    beta2=beta*beta;beta3=beta2*beta
    nonself=!((shift==Int32(22))&(jy<=ix))
    j4=jstart
    @inbounds while j4<jend
        base=Int32(8)*j4
        imask=cj[base+Int32(5)+Int32(2)*warp]
        if imask==UInt32(0);j4+=Int32(1);continue;end
        exclIndex=Int32(cj[base+Int32(6)+Int32(2)*warp])
        exclWord=ex[Int32(32)*exclIndex+(tid&Int32(31))+Int32(1)]
        for jm in Int32(0):Int32(3)
            if (imask&(UInt32(0xff)<<(jm*Int32(8))))==UInt32(0);continue;end
            jcluster=Int32(cj[base+jm+Int32(1)]);atomJ=jcluster*Int32(8)+jy
            qj=nb_load4(xq,atomJ);typeJ=types[atomJ+Int32(1)]
            fxJ=0f0;fyJ=0f0;fzJ=0f0
            Base.Cartesian.@nexprs 8 c -> begin
                ciOffset=Int32(c-1)
                bit=UInt32(1)<<(jm*Int32(8)+ciOffset)
                if (imask&bit)!=UInt32(0)
                    clusterI=sci*Int32(8)+ciOffset
                    a=Int32(4)*(ciOffset*Int32(8)+ix)
                    dx=smx[a+Int32(1)]-qj[1]
                    dy=smx[a+Int32(2)]-qj[2]
                    dz=smx[a+Int32(3)]-qj[3]
                    r2=muladd(dz,dz,muladd(dy,dy,dx*dx))
                    notExcluded=nonself|(clusterI!=jcluster)
                    if r2<rc2 && notExcluded
                        qi=smx[a+Int32(4)]
                        pairMask=ifelse((exclWord&bit)!=UInt32(0),1f0,0f0)
                        typeI=smt[ciOffset*Int32(8)+ix+Int32(1)]
                        co=Int32(2)*(numtypes*typeI+typeJ)
                        c6=coeff[co+Int32(1)];c12=coeff[co+Int32(2)]
                        r2=max(r2,3.82f-7)
                        rinv=nb_rsqrt_rn(r2);r2inv=rinv*rinv
                        r6inv=(r2inv*r2inv)*r2inv;r6inv*=pairMask
                        scale=(r6inv*muladd(c12,r6inv,-c6))*r2inv
                        coulomb=muladd(pairMask*r2inv,rinv,nb_pme(beta2*r2)*beta3)
                        scale=muladd(qi*qj[4],coulomb,scale)
                        # Explicit FMA for multiply/add contraction as in the source
                        # GPU compilation; actual instruction equivalence unprofiled.
                        fxJ=muladd(-dx,scale,fxJ);fyJ=muladd(-dy,scale,fyJ);fzJ=muladd(-dz,scale,fzJ)
                        fxI=Base.setindex(fxI,muladd(dx,scale,fxI[c]),c)
                        fyI=Base.setindex(fyI,muladd(dy,scale,fyI[c]),c)
                        fzI=Base.setindex(fzI,muladd(dz,scale,fzI[c]),c)
                    end
                end
            end
            nb_reduce_j!(fxJ,fyJ,fzJ,ix,atomJ,f)
        end
        j4+=Int32(1)
    end
    (fxI,fyI,fzI)
end
