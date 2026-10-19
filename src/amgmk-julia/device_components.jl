# Native in-place relaxation; no Jacobi/coloring/atomic substitution.
@inline function amg_relax(i::Int32,n::Int32,data,idx,cols,u,f)
 if i<n
  @inbounds beg=idx[i+Int32(1)];en=idx[i+Int32(2)];diag=data[beg+Int32(1)]
  if diag!=0.0
   @inbounds res=f[i+Int32(1)]
   for j in beg+Int32(1):en-Int32(1)
    @inbounds col=cols[j+Int32(1)];res-=data[j+Int32(1)]*u[col+Int32(1)]
   end
   @inbounds u[i+Int32(1)]=res/diag
  end
 end
 nothing
end
