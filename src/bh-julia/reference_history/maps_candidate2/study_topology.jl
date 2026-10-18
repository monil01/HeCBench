function bh_treecheck(n,nn,p,order,counts,starts,children,bottom)
 lo=Int(bottom);nn=Int(nn);n=Int(n);n<=lo<=nn||return false,"bottom_range",0,0
 nodes=Set{Int}();leaves=Set{Int}();depthmax=Ref(0);reason=Ref("");dfs=Int[]
 function visit(node,depth)
  if depth>31;reason[]="stack_depth";return 0,0.;end
  if node<n
   if !(0<=node<n)||node in leaves;reason[]="duplicate_or_invalid_leaf";return 0,0.;end
   push!(leaves,node);push!(dfs,node);return 1,Float64(p[4node+4])
  end
  if !(lo<=node<=nn)||node in nodes;reason[]="invalid_or_repeated_cell";return 0,0.;end
  push!(nodes,node);depthmax[]=max(depthmax[],depth);start=length(dfs);count=0;mass=0.;seenempty=false
  for j in 0:7
   ch=Int(children[8node+j+1])
   if ch == -1;seenempty=true
   elseif ch<0||seenempty;reason[]="locked_or_uncompacted_child";return 0,0.
   else;ct,m=visit(ch,depth+1);count+=ct;mass+=m;isempty(reason[])||return 0,0.;end
  end
  if count==0||counts[node+1]!=count||starts[node+1]!=start;reason[]="subtree_count_or_start";return 0,0.;end
  if !all(isfinite,p[4node+1:4node+4])||abs(Float64(p[4node+4])-mass)>1e-6;reason[]="cell_mass_or_finite_com";return 0,0.;end
  return count,mass
 end
 count,mass=visit(nn,0)
 ok=isempty(reason[])&&count==n&&length(leaves)==n&&length(nodes)==nn-lo+1&&dfs==Int.(order[1:n])&&abs(mass-1)<=1e-6
 return ok,isempty(reason[]) ? (ok ? "ok" : "coverage_order_mass") : reason[],length(nodes),depthmax[]
end
