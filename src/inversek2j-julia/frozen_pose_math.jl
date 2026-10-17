# Native inversek2j executable behavior, NOT a valid CCD solver.
# Source author Amir Yazdanbakhsh, Alternative Computing Technologies Lab., Georgia Tech,2015.
# Preserve frozen initial geometry, degree-clamped assignments and repeated prefix sums.
@inline function frozen_pose_angles(x,y)
 a0=0f0;a1=0f0;a2=0f0
 for loop in 1:25
  for it in 3:-1:1
   dx1=Float32(4-it);dy1=0f0;dx2=x-Float32(it-1);dy2=y
   l1=sqrt(dx1*dx1+dy1*dy1);l2=sqrt(dx2*dx2+dy2*dy2)
   ax=dx1/l1;ay=dy1/l1;bx=dx2/l2;by=dy2/l2
   dot=ax*bx+ay*by
   if dot>1f0;dot=1f0;elseif dot< -1f0;dot= -1f0;end
   angle=acos(dot)*(180f0/3.14159265358979f0)
   if ax*by-ay*bx<0f0;angle= -angle;end
   if angle>30f0;angle=30f0;elseif angle< -30f0;angle= -30f0;end
   if it==1;a0=angle;elseif it==2;a1=angle;else;a2=angle;end
   a1+=a0;a2+=a1
  end
 end
 (a0,a1,a2)
end
