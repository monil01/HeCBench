# Source-derived compact material arithmetic. Index encodings remain source zero-based.
@inline function fracstart(d,c,::Val{M}) where M
 ix=d.imaterial[c]
 return M==2 ? -ix : d.mmc_index[-ix+1]
end
@inline function fracnext(d,c,ix,::Val{M}) where M
 if M==2
  return d.nextfrac[ix+1]
 else
  return ix+1 < d.mmc_index[-d.imaterial[c]+2] ? ix+1 : Int32(-1)
 end
end
@inline function density_cell!(d,o,i,j,mode::Val{M}) where M
 c=i+96*j+1
 ave=0.0
 if M!=0 && d.imaterial[c]<=0
  ix=fracstart(d,c,mode)
  while ix>=0
   ave+=d.rho_list[ix+1]*d.vf_list[ix+1]
   ix=fracnext(d,c,ix,mode)
  end
 else
  ave=d.rho_cell[c]
 end
 o.rho_ave[c]=ave/d.V[c]
 return nothing
end
@inline function density_mixed!(d,o,c)
 ave=0.0
 for ix=d.mmc_index[c+1]:d.mmc_index[c+2]-1
  ave+=d.rho_list[ix+1]*d.vf_list[ix+1]
 end
 k=d.mmc_i[c+1]+96*d.mmc_j[c+1]+1
 o.rho_ave[k]=ave/d.V[k]
 return nothing
end
@inline function pressure_cell!(d,o,i,j,mode::Val{M}) where M
 c=i+96*j+1;im=d.imaterial[c]
 if im<=0
  if M!=0
   ix=fracstart(d,c,mode)
   while ix>=0
    k=ix+1;m=d.matids[k]+1
    o.p_list[k]=((d.n[m]*d.rho_list[k])*d.t_list[k])/d.vf_list[k]
    ix=fracnext(d,c,ix,mode)
   end
  end
 else
  o.p_cell[c]=(d.n[im]*d.rho_cell[c])*d.t_cell[c]
 end
 return nothing
end
@inline function pressure_fraction!(d,o,ix)
 k=ix+1;m=d.matids[k]+1
 o.p_list[k]=((d.n[m]*d.rho_list[k])*d.t_list[k])/d.vf_list[k]
 return nothing
end
@inline function neighbor_sum(d,c,i,j,mat,dist,mode)
 total=0.0;nn=Int32(0)
 for dj=-1:1
  for di=-1:1
   if di==0 && dj==0
    continue # explicit corrected-reference self exclusion, original is nonfinite
   end
   q=(i+di)+96*(j+dj)+1;im=d.imaterial[q]
   if im<=0
    ix=fracstart(d,q,mode)
    while ix>=0
     if d.matids[ix+1]==mat
      total+=d.rho_list[ix+1]/dist[(dj+1)*3+di+2];nn+=1
      break
     end
     ix=fracnext(d,q,ix,mode)
    end
   elseif mat==im-1
    total+=d.rho_cell[q]/dist[(dj+1)*3+di+2];nn+=1
   end
  end
 end
 return total/nn
end
@inline function neighbors_cell!(d,o,i,j,mode)
 c=i+96*j+1;xo=d.x[c];yo=d.y[c]
 dist=ntuple(Val(9)) do k
  di=(k-1)%3-1;dj=(k-1)÷3-1;q=i+di+96*(j+dj)+1
  dx=xo-d.x[q];dy=yo-d.y[q]
  dx*dx+dy*dy
 end
 im=d.imaterial[c]
 if im<=0
  ix=fracstart(d,c,mode)
  while ix>=0
   o.neighbor_list[ix+1]=neighbor_sum(d,c,i,j,d.matids[ix+1],dist,mode)
   ix=fracnext(d,c,ix,mode)
  end
 else
  o.neighbor_cell[c]=neighbor_sum(d,c,i,j,im-1,dist,mode)
 end
 return nothing
end
