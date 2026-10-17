# Native insertion-topk algorithm, translated once; stable ascending reference-index ties.
@inline function study_insert!(col,dist,ind,width,height,k)
 if col<=width
  maxdist=dist[col];ind[col]=Int32(0)
  for l in 1:k-1
   row=l*width;curr=dist[row+col]
   if curr<maxdist
    at=l-1
    for a in 0:l-2
     if dist[a*width+col]>curr;at=a;break;end
    end
    for j in l:-1:at+1;dist[j*width+col]=dist[(j-1)*width+col];ind[j*width+col]=ind[(j-1)*width+col];end
    dist[at*width+col]=curr;ind[at*width+col]=Int32(l)
   else
    ind[row+col]=Int32(l)
   end
   maxdist=dist[row+col]
  end
  maxrow=(k-1)*width
  for l in k:height-1
   curr=dist[l*width+col]
   if curr<maxdist
    at=k-1
    for a in 0:k-2
     if dist[a*width+col]>curr;at=a;break;end
    end
    for j in k-1:-1:at+1;dist[j*width+col]=dist[(j-1)*width+col];ind[j*width+col]=ind[(j-1)*width+col];end
    dist[at*width+col]=curr;ind[at*width+col]=Int32(l);maxdist=dist[maxrow+col]
   end
  end
 end
 nothing
end
