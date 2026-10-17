# Six source mask predicates, translated once and specialized by Val peroperator.
# Copyright/origin and redistribution terms: LICENSE inthisbenchmarkparent, preservedfrommask-cuda.
@inline function study_keep(::Val{M},j,k,seq,centers,radius) where M
 if M==1
  k<seq[j+1]
 elseif M==2
  centers[j+1]-radius<=k<=centers[j+1]+radius
 elseif M==3
  k<=j
 elseif M==4
  k>=j
 elseif M==5
  k<j
 else
  k>j
 end
end
