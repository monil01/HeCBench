from pathlib import Path
import json,struct,math
p=Path(__file__).resolve().parent;U=lambda x:struct.unpack('<I',struct.pack('<f',x))[0];F=lambda x:struct.unpack('<f',struct.pack('<I',x))[0];pre=[5];gold=[];cases=[];guard=-12345.;gu=U(guard)
for ci,(W,H) in enumerate([(17,13),(19,11),(1,5),(7,1),(9,7)]):
 pts=[]
 for y in range(H):
  for x in range(W):
   z=1+(3*x+5*y)/1024 if ci!=4 else -1+(3*x+5*y)/1024
   if ci==1:z=1 if x<W//2 else 201 # exact200jump rejectedstrict<200
   pts.append([U(x*.25),U(y*.25),U(z)])
 if ci==1:
  for index,bits in [(0,0x7fc00011),(W+2,0xffc00005),(5*W+10,0x7fc00023)]:pts[index][2]=bits
 a=[[F(v) for v in q] for q in pts];out=[];status=[];normalcount=0;minnorm=math.inf;maxplane=0
 for idx,q in enumerate(a):
  if math.isnan(q[2]):out.extend([0,0,0,0]);status.append('invalid_query');continue
  x=idx%W;y=idx//W
  valid=[x>1 and not math.isnan(a[idx-1][2]) and abs(a[idx-1][2]-q[2])<200,x<W-1 and not math.isnan(a[idx+1][2]) and abs(a[idx+1][2]-q[2])<200,y>1 and not math.isnan(a[idx-W][2]) and abs(a[idx-W][2]-q[2])<200,y<H-1 and not math.isnan(a[idx+W][2]) and abs(a[idx+W][2]-q[2])<200]
  west,east,north,south=valid
  if not(west or east) or not(north or south):out.extend([0,0,0,1]);status.append('insufficient_neighbors');continue
  sub=lambda i,j:[a[i][k]-a[j][k] for k in range(3)]
  h=sub(idx+1,idx-1) if west and east else sub(idx,idx-1) if west else sub(idx+1,idx)
  v=sub(idx-W,idx+W) if north and south else sub(idx,idx+W) if south else sub(idx-W,idx)
  n=[h[1]*v[2]-h[2]*v[1],h[2]*v[0]-h[0]*v[2],h[0]*v[1]-h[1]*v[0]];norm=math.sqrt(sum(t*t for t in n));assert norm>0;minnorm=min(minnorm,norm);n=[z/norm for z in n]
  if sum(q[k]*n[k] for k in range(3))>0:n=[-z for z in n]
  flag=abs(h[2])>float(F(U(.04))) or abs(v[2])>float(F(U(.04))) or not all(valid);out.extend(n+[int(flag)]);status.append('normal');normalcount+=1
  if ci in [0,4]:
   analytic=[3/256,5/256,-1];an=math.sqrt(sum(z*z for z in analytic));analytic=[z/an*(1 if ci==0 else -1) for z in analytic];maxplane=max(maxplane,max(abs(x-y) for x,y in zip(n,analytic)))
 pre +=[W,H]+[v for q in pts for v in q]
 gold +=[guard]*(16*4)+out+[guard]*(8*4)
 gold +=[gu]*(64*3)+[v for q in pts for v in q]+[gu]*(8*3)
 cases.append(dict(case=ci,width=W,height=H,points=W*H,status=status,normal_outputs=normalcount,min_unnormalized_cross_norm=minnorm if normalcount else None,analytic_plane_normal_max_error=maxplane if ci in [0,4] else None,invalid_z_nan_payload_bits=[hex(q[2]) for q in pts if math.isnan(F(q[2]))],strict_depth_jump=200 if ci==1 else None))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),atol=1e-6,rtol=3e-6,oracle='IndependentFloat64cross/normalize/cameraorientation usingactualF32points/sourceexactneighborpredicates, analyticplanes supplement. Exact flag/invalid outputs, allinputNaNpayloadbits andguards. Degeneratefinitecross0 isoutsideacceptedinputdomain, noNaNoutputaccepted.',layout='Nativefloat3 AoS12bytes, float4AoS16bytes; JuliaisbitsPoint3size12 and4VecElementF32tuple16. Inputleading64Point3/outleading16Float4 preserve256Bpayloadalignment, trailing8elements. Inputwords encodedUInt32 losslesslyasF64numericwords, notNaNfloatcanonicalization.',source_semantics='westx>1/northy>1 asymmetryretained, depthjumpstrict<200, curvaturew isBooleanedge/discontinuityflagnotdifferentialcurvature. Normal faces camera ifdot(query,normal)>0 thenflip.'),indent=2)+'\n');print(len(pre),len(gold))
