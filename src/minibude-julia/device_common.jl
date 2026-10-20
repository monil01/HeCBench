# Source Atom/FFParams are packed 16-byte records with four 4-byte fields.
struct BudeAtom
    x::Float32
    y::Float32
    z::Float32
    type::Int32
end
struct BudeFF
    hbtype::Int32
    radius::Float32
    hphb::Float32
    elsc::Float32
end
@assert sizeof(BudeAtom)==16 && sizeof(BudeFF)==16
@assert fieldoffset(BudeAtom,4)==12 && fieldoffset(BudeFF,4)==12

@inline function prepare_transform(t0,t1,t2,t3,t4,t5,ix::Int32)
    @inbounds begin
        sx=sin(t0[ix+Int32(17)]);cx=cos(t0[ix+Int32(17)])
        sy=sin(t1[ix+Int32(17)]);cy=cos(t1[ix+Int32(17)])
        sz=sin(t2[ix+Int32(17)]);cz=cos(t2[ix+Int32(17)])
        return (cy*cz,sx*sy*cz-cx*sz,cx*sy*cz+sx*sz,t3[ix+Int32(17)],
                cy*sz,sx*sy*sz+cx*cz,cx*sy*sz-sx*cz,t4[ix+Int32(17)],
                -sy,sx*cy,cx*cy,t5[ix+Int32(17)])
    end
end

# Compile-time source NUM_TD_PER_THREAD=1: private arrays each have one element.
# All padded lanes evaluate the clamped final pose, then only valid lanes store.
@inline function evaluate_pose!(protein,ligand,cache,out,T,n,nl,np,baseid)
    energy=0f0;il=Int32(0)
    @inbounds while il<nl
        atom=ligand[il+Int32(17)];lp=cache[atom.type+Int32(1)]
        lneg=lp.hphb<0f0;lpos=lp.hphb>0f0
        x0=T[4]+atom.x*T[1]+atom.y*T[2]+atom.z*T[3]
        y0=T[8]+atom.x*T[5]+atom.y*T[6]+atom.z*T[7]
        z0=T[12]+atom.x*T[9]+atom.y*T[10]+atom.z*T[11]
        ip=Int32(0)
        while ip<np
            p=protein[ip+Int32(17)];pp=cache[p.type+Int32(1)]
            radius=pp.radius+lp.radius;rradius=1f0/radius
            bothF=pp.hbtype==Int32(70)&&lp.hbtype==Int32(70)
            cutoff=ifelse(bothF,4f0,2f0);rcutoff=ifelse(bothF,.25f0,.5f0)
            typeE=pp.hbtype==Int32(69)||lp.hbtype==Int32(69)
            pneg=pp.hphb<0f0;ppos=pp.hphb>0f0;pnz=pp.hphb!=0f0
            ph=pp.hphb*ifelse(pneg&&lpos,-1f0,1f0)
            lh=lp.hphb*ifelse(ppos&&lneg,-1f0,1f0)
            length=ifelse(pneg,ifelse(lneg,5.5f0,1f0),ifelse(lneg,1f0,-floatmax(Float32)))
            rlength=1f0/length;charge0=lp.elsc*pp.elsc;hydro0=ph+lh
            x=x0-p.x;y=y0-p.y;z=z0-p.z
            distance=sqrt(x*x+y*y+z*z);surface=distance-radius;overlap=surface<0f0
            energy+=(1f0-distance*rradius)*ifelse(overlap,76f0,0f0)
            charge=charge0*(ifelse(overlap,1f0,1f0-surface*rcutoff)*ifelse(surface<cutoff,1f0,0f0))
            charge=ifelse(typeE,-abs(charge),charge)
            energy+=charge*45f0
            coeff=1f0-surface*rlength
            hydro=hydro0*ifelse(surface<length&&pnz,1f0,0f0)
            hydro*=ifelse(overlap,1f0,coeff)
            energy+=hydro
            ip+=Int32(1)
        end
        il+=Int32(1)
    end
    if baseid<n
        @inbounds out[baseid+Int32(17)]=energy*.5f0
    end
    nothing
end
