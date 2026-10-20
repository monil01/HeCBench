# Source-mapping diagnostic, not an independent mathematical oracle.
include(joinpath(@__DIR__,"study_device.jl"))
@inline snap_atomic!(a,i,v)=(a[i]+=v;nothing)
for J in (2,4)
 open(joinpath(@__DIR__,"data/J$(J)_source_fixture.bin")) do io
  rd(T,n)=read!(io,Vector{T}(undef,n));h=rd(UInt32,12);J,na,ng,nb,ncoeff,nu,nd,nz,ncg,jdim,pdim=Int32.(h[2:12]);s=rd(Float64,4)
  iu=rd(Int32,jdim);parity=rd(Int32,nu);id=rd(Int32,jdim);iz=rd(Int32,9nz);icg=rd(Int32,jdim^3);root=rd(Float64,pdim^2);cg=rd(Float64,ncg);zb=rd(Float64,nz);r=rd(Float64,3na*nb);inside=rd(Int32,na*nb);w=rd(Float64,na*nb);rc=rd(Float64,na*nb)
  u=zeros(Float64,2na*nb*nu);ut=zeros(Float64,2na*nu);y=zeros(Float64,2na*nd);du=zeros(Float64,6na*nb*nd);d=zeros(Float64,3na*nb)
  for a in Int32(0):na-1;set_ulisttot(ut,iu,na,J,s[2],a,Int32(0));end
  for b in Int32(0):nb-1,a in Int32(0):na-1;update_ulisttot(r,rc,w,parity,iu,root,u,ut,na,nb,Int32(1),J,pdim,a,b);end
  for z in Int32(0):nz-1,a in Int32(0):na-1;compute_yi(iz,zb,cg,icg,iu,id,ut,y,na,nz,jdim,a,z);end
  for b in Int32(0):nb-1,a in Int32(0):na-1;compute_duidrj(w,r,rc,root,u,du,na,nb,J,nd,pdim,Int32(1),a,b);compute_deidrj(id,du,y,d,na,nb,J,nd,a,b);end
  want=collect(reinterpret(Float64,read(joinpath(@__DIR__,"data/J$(J)_independent_oracle.bin"))))[1:length(d)]
  err=maximum(abs.(d.-want));ratio=maximum(abs.(d.-want)./(1e-7.+1e-9.*abs.(want)));println("CPU_TRANSLATION J=",J," max_abs=",err," max_budget_ratio=",ratio);ratio<=1||error("source-mapping check")
 end
end
