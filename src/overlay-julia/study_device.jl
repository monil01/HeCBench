struct OverlayBox
 width::Int32
 height::Int32
 left::Int32
 top::Int32
end
@inline function overlay_pixel(input,output,W,H,x0,y0,bw,bh,color,bx,by)
 if bx<bw&&by<bh
  x=bx+x0;y=by+y0
  if x<W&&y<H
   index=y*W+x+Int32(1)
   @inbounds pixel=input[index]
   alpha=color[4]/255f0;ialph=1f0-alpha
   # Preserve source algebra; no fast-math option. FMA contraction may differ.
   value=(alpha*color[1]+ialph*pixel[1],alpha*color[2]+ialph*pixel[2],alpha*color[3]+ialph*pixel[3])
   @inbounds output[index]=value
  end
 end
 return
end
