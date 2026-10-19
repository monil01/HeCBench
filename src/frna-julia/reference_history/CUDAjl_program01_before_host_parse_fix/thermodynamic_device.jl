# Julia translation of original FRNA thermodynamic helpers (GPL source retained).
using CUDA
const RNA_INF=Int32(14000)
const RNA_A=Int32(0); const RNA_C=Int32(1); const RNA_G=Int32(2); const RNA_U=Int32(3)
struct RNAArray{T,A}
    data::A
    offset::Int32
end
RNAArray{T}(a::A,offset::Int32=Int32(0)) where {T,A}=RNAArray{T,A}(a,offset)
@inline Base.getindex(a::RNAArray{T},i) where T = @inbounds reinterpret(T,a.data[a.offset+Int32(i)+Int32(1)])
@inline function Base.setindex!(a::RNAArray{T},x,i) where T
    @inbounds a.data[a.offset+Int32(i)+Int32(1)]=reinterpret(eltype(a.data),T(x))
    x
end
@inline rna_offset(a::RNAArray{T},i) where T = RNAArray{T}(a.data,a.offset+Int32(i))
struct RNARef{P}
    pointer::P
    valid::Bool
end
@inline rna_invalid(a)=RNARef(a,false)
@inline rna_truth(x::Bool)=x
@inline rna_truth(x::Integer)=x!=0
@inline rna_truth(x::RNARef)=x.valid
@inline rna_deref(x::RNARef)=x.pointer[Int32(0)]
struct RNATable{N,A}
    data::A
    offset::Int32
end
RNATable{N}(a::A,offset::Int32) where {N,A}=RNATable{N,A}(a,offset)
@inline Base.getindex(t::RNATable{1},i)=@inbounds reinterpret(Int32,t.data[t.offset+Int32(i)+Int32(1)])
@inline Base.getindex(t::RNATable{N},i) where N=RNATable{N-1}(t.data,t.offset+Int32(i)*Int32(4^(N-1)))
struct RNALoops{W,A}
    data::A
    offset::Int32
end
RNALoops{W}(a::A,offset::Int32) where {W,A}=RNALoops{W,A}(a,offset)
struct RNALoop{W,A}
    data::A
    offset::Int32
end
@inline Base.getindex(t::RNALoops{W,A},i) where {W,A}=RNALoop{W,A}(t.data,t.offset+Int32(i)*Int32(W+1))
@inline function Base.getproperty(t::RNALoop{W},name::Symbol) where W
    a=getfield(t,:data);offset=getfield(t,:offset)
    name===:seq && return RNAArray{Int32}(a,offset)
    @inbounds reinterpret(Int32,a[offset+Int32(W)+Int32(1)])
end
struct RNAParams{A}
    data::A
end
include(joinpath(@__DIR__,"parameter_accessors.jl"))
@inline is_cp(i,j)=(i==RNA_A&&j==RNA_U)||(i==RNA_C&&j==RNA_G)||(i==RNA_G&&j==RNA_U)
@inline is_canonical_pair(i,j)=is_cp(i,j)||is_cp(j,i)
@inline cp(i,j,s)=j-i-Int32(1)>=3 && is_canonical_pair(s[i],s[j])
@inline function can_pair(i,j,n,s)
    if j<i;i,j=j,i;end
    cp(i,j,s)&&((i>0&&j<n-Int32(1)&&cp(i-Int32(1),j+Int32(1),s))||cp(i+Int32(1),j-Int32(1),s))
end
@inline function not_isolated(i,j,n,s)
    if j<i;i,j=j,i;end
    is_canonical_pair(s[i],s[j])&&((i>0&&j<n-Int32(1)&&cp(i-Int32(1),j+Int32(1),s))||cp(i+Int32(1),j-Int32(1),s))
end
@inline wrap(i,n)=i>=n ? i-n : i
@inline is_exterior(i,j)=j<i
@inline is_interior(i,j)=i<j
@inline ind(i,j,n)=i*n+j
@inline function array_val(a,i,j,n,s)
    valid=can_pair(i,j,n,s)
    RNARef(rna_offset(a,valid ? ind(i,j,n) : Int32(0)),valid)
end
@inline function contains_only_base(b,n,s)
    for i in Int32(0):n-Int32(1)
        if s[i]!=b; return false;end
    end
    true
end
@inline function sequences_match(a,b,n)
    for i in Int32(0):n-Int32(1)
        if a[i]!=b[i];return false;end
    end
    true
end
@inline terminal_U_penalty(s,i,j,p)=(s[i]==RNA_U||s[j]==RNA_U) ? p.terminal_AU_penalty : Int32(0)
@inline dangle_3p_energy(s,i,j,ip1,p)=p.dangle_3p[s[i]][s[j]][s[ip1]]+terminal_U_penalty(s,i,j,p)
@inline dangle_5p_energy(s,i,j,jm1,p)=p.dangle_5p[s[i]][s[j]][s[jm1]]+terminal_U_penalty(s,i,j,p)
@inline terminal_stack(s,i,j,ip1,jm1,p)=p.tstack[s[i]][s[j]][s[ip1]][s[jm1]]+terminal_U_penalty(s,i,j,p)
@inline terminal_stack_multibranch(s,i,j,ip1,jm1,p)=p.tstackm[s[i]][s[j]][s[ip1]][s[jm1]]+terminal_U_penalty(s,i,j,p)
@inline function rna_special_loop(s,d,p)
    if d==3
        for i in Int32(0):p.ntriloop-Int32(1)
            entry=p.triloop[i]
            if sequences_match(s,entry.seq,Int32(5));return true,entry.val;end
        end
    elseif d==4
        for i in Int32(0):p.ntloop-Int32(1)
            entry=p.tloop[i]
            if sequences_match(s,entry.seq,Int32(6));return true,entry.val;end
        end
    elseif d==6
        for i in Int32(0):p.nhexaloop-Int32(1)
            entry=p.hexaloop[i]
            if sequences_match(s,entry.seq,Int32(8));return true,entry.val;end
        end
    end
    false,Int32(0)
end
@inline function hairpin_loop_energy(s,i,j,d,p)
    found,value=rna_special_loop(rna_offset(s,i),d,p)
    if found;return value;end
    if d>30
        x=Base.FastMath.log_fast(Float32(d)/30f0)
        e=unsafe_trunc(Int32,muladd(p.prelog,x,Float32(p.hairpin_loop_initiation[Int32(30)])))
    else
        e=p.hairpin_loop_initiation[d]
    end
    allc=contains_only_base(RNA_C,d,rna_offset(s,i+Int32(1)))
    if d==3
        if allc;e+=p.c_hairpin_of_3;end
        e+=terminal_U_penalty(s,i,j,p)
    else
        e+=p.tstackh[s[i]][s[j]][s[i+Int32(1)]][s[j-Int32(1)]]
        if allc;e+=p.c_hairpin_slope*d+p.c_hairpin_intercept;end
    end
    if s[i]==RNA_G&&s[j]==RNA_U&&i>1&&s[i-Int32(1)]==RNA_G&&s[i-Int32(2)]==RNA_G
        e+=p.bonus_for_GGG_hairpin
    end
    e
end
@inline function alternative_bulge_loop_correction(n,s,i,ip)
    count=Int32(1)
    if i!=n-Int32(1)
        k=i
        while k>=0&&s[k]==s[i+Int32(1)];count+=Int32(1);k-=Int32(1);end
        k=ip
        while k<=n-Int32(1)&&s[k]==s[i+Int32(1)];count+=Int32(1);k+=Int32(1);end
    end
    # RT is a source Double constant; log argument and result are Float32.
    unsafe_trunc(Int32,(-1.0*(1.987213e-3*310.15)*Float64(100f0))*Float64(Base.FastMath.log_fast(Float32(count))))
end
@inline function internal_loop_energy(s,n,i,j,ip,jp,d1,d2,p)
    if d1==0||d2==0
        e=p.bulge_loop_initiation[d1+d2]
        if d1==1||d2==1
            e+=p.stack[s[i]][s[j]][s[ip]][s[jp]]
            e+=d1==0 ? alternative_bulge_loop_correction(n,s,jp,j) : alternative_bulge_loop_correction(n,s,i,ip)
            if (d1==1&&s[i+Int32(1)]==RNA_C&&(s[i]==RNA_C||s[i+Int32(2)]==RNA_C))||(d2==1&&s[j-Int32(1)]==RNA_C&&(s[j]==RNA_C||s[j-Int32(2)]==RNA_C))
                e+=p.Bonus_for_Single_C_bulges_adjacent_to_C
            end
        else
            e+=terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)
        end
        return e
    end
    if d1==1&&d2==1
        return p.int11[s[i]][s[i+Int32(1)]][s[i+Int32(2)]][s[j-Int32(2)]][s[j-Int32(1)]][s[j]]
    elseif d1==2&&d2==2
        return p.int22[s[i]][s[ip]][s[j]][s[jp]][s[i+Int32(1)]][s[i+Int32(2)]][s[j-Int32(1)]][s[j-Int32(2)]]
    elseif d1==1&&d2==2
        return p.int21[s[i]][s[j]][s[i+Int32(1)]][s[j-Int32(1)]][s[jp+Int32(1)]][s[ip]][s[jp]]
    elseif d1==2&&d2==1
        return p.int21[s[jp]][s[ip]][s[jp+Int32(1)]][s[ip-Int32(1)]][s[i+Int32(1)]][s[j]][s[i]]
    end
    sp=(d1==1||d2==1) ? p.tstacki1n : ((d1==2&&d2==3)||(d1==3&&d2==2)) ? p.tstacki23 : p.tstacki
    p.internal_loop_initiation[d1+d2]+min(p.fm_array_first_element*abs(d1-d2),p.maximum_correction)+sp[s[i]][s[j]][s[i+Int32(1)]][s[j-Int32(1)]]+sp[s[jp]][s[ip]][s[jp+Int32(1)]][s[ip-Int32(1)]]
end
@inline coaxial_flush(s,i,j,ip,jp,p)=terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+p.coaxial[s[i]][s[j]][s[ip]][s[jp]]
@inline coaxial_mismatch1(s,i,j,ip,jp,p)=terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+p.tstackcoax[s[j]][s[i]][s[j+Int32(1)]][s[i-Int32(1)]]+p.coaxstack[s[j+Int32(1)]][s[i-Int32(1)]][s[ip]][s[jp]]
@inline coaxial_mismatch2(s,i,j,ip,jp,p)=terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+p.tstackcoax[s[jp]][s[ip]][s[jp+Int32(1)]][s[ip-Int32(1)]]+p.coaxstack[s[j]][s[i]][s[j+Int32(1)]][s[jp+Int32(1)]]
include(joinpath(@__DIR__,"recurrence_bodies.jl"))
