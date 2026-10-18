# Same packed-upper indexing as original native ker and ker2.
@inline function extract_site!(C,U,idx,n1,n)
 z=idx-1
 if z<n1*n
  i=z÷n;j=z%n
  if i<j&&i<n1
   t=n*i-i*(i+1)÷2+j-i
   @inbounds U[t]=C[j*n1+i+1]
  end
 end
 nothing
end
