"""Exact dyadic/integer integral inputs, independent raw-pixel box oracle; CPU only."""
from pathlib import Path
import json,math,struct,hashlib
p=Path(__file__).resolve().parent;cases=[]
for h,w,kind in [(17,31,'nonnegative_texture'),(32,16,'positive_impulses_boundaries'),(9,33,'signed_operator_control'),(1,19,'single_row_boundary_control')]:
 raw=[((r*13+c*7+3*r*c)%16) if kind=='nonnegative_texture' else ((r*11+c*5)%13-6) if kind=='signed_operator_control' else 2 for r in range(h) for c in range(w)]
 if kind=='positive_impulses_boundaries':
  for r,c in [(0,0),(0,w//2),(h//2,0),(h//2,w//2),(h-1,w-1)]:raw[r*w+c]+=40
 integ=[]
 for r in range(h):
  for c in range(w):integ.append(sum(raw[y*w+x] for y in range(r+1) for x in range(c+1)))
 assert 4*sum(abs(x) for x in raw)<2**24
 cases.append((h,w,kind,raw,integ))
sigmas=[1.,3.,4.,4.25];prefix=[]
for h,w,kind,raw,integ in cases:prefix.extend(map(float,raw));prefix.extend(map(float,integ))
for h,w,*_ in cases:prefix.extend([float(h),float(w)])
prefix.extend(sigmas);(p/'fixture.bin').write_bytes(struct.pack('<'+'d'*len(prefix),*prefix));gold=[];stats=[]
for ci,(h,w,kind,raw,integ) in enumerate(cases):
 def rect(r,c,rl,cl):
  rr=min(max(r,0),h-1);cc=min(max(c,0),w-1);r2=min(rr+rl,h-1);c2=min(cc+cl,w-1)
  return max(0,sum(raw[y*w+x] for y in range(rr+1,r2+1) for x in range(cc+1,c2+1)))
 for sigma in sigmas:
  size=int(3*sigma);b=(size-1)//2+1;l=size//3;wi=1/(size*size);out=[]
  for r in range(h):
   for c in range(w):
    tl=rect(r-l,c-l,l,l);br=rect(r+1,c+1,l,l);bl=rect(r-l,c+1,l,l);tr=rect(r+1,c-l,l,l)
    dxy=-(bl+tr-tl-br)*wi
    dxx=-(rect(r-l+1,c-l,2*l-1,size)-3*rect(r-l+1,c-l//2,2*l-1,l))*wi
    dyy=-(rect(r-l,c-b+1,size,2*b-1)-3*rect(r-b//2,c-b+1,b,2*b-1))*wi
    out.append(dxx*dyy-.81*(dxy*dxy))
  gold.extend([-12345.]*8+out+[-12345.]*8);gold.extend([-12345.]*8+list(map(float,integ))+[-12345.]*8)
  stats.append({'image':ci,'kind':kind,'h':h,'w':w,'sigma_Float32':sigma,'size_trunc_3sigma':size,'payload_min':min(out),'payload_max':max(out),'zero_outputs':sum(x==0 for x in out)})
(p/'oracle.bin').write_bytes(struct.pack('<'+'d'*len(gold),*gold));meta={'cases':[{'h':h,'w':w,'kind':kind,'pixels':h*w,'raw_and_integral_integer_exact_bound':4*sum(abs(x) for x in raw)} for h,w,kind,raw,integ in cases],'sigmas':sigmas,'paths':stats,'prefix_count':len(prefix),'output_count':len(gold),'artifact_count':len(prefix)+len(gold),'input_sha256':hashlib.sha256((p/'fixture.bin').read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle.bin').read_bytes()).hexdigest(),'precision':'ActualGPUintegralFloat32; raw/prefix/sums bounded exactintegers (<2^24 even cornerintermediates), so originalnaiveF32integral andgenuineJuliafasterprefixprepareidenticalselecteddata. FinalGPUoutputFloat32 losslesslywidenedFloat64 forartifact.','independent_oracle':'Directraw-pixelsums, notintegraltablecornerlookup, with exactsourceclippedstart-before-end andmax0 semantics, idealFloat64boxfilteralgebra. ActualsourceFloat32normalization/coefficient/FMA/grouping allowedonlypredeclaredatol1e-4+rtol2e-6.','boundary_policy':'InclusiveintegralI[r,c]=sum raw[0:r,0:c]; sourcecornerexpression denotes raw rows(r,r2],cols(c,c2], afterclippingstart then addingextent. Firstrawrow/colneverincluded. Thisis unchangedsourceoperatorpolicy, NOTidealzeropadding/continuousHessian accuracy.','scientific_scope':'Box-filterapproximate Hessian determinant; sourcewarnssigma<3 approximatequality poor. sigma1/signed/singlerow are operatorcontrols, notGaussianderivativequalityclaims;4/4.25 samequantizedsize12 mustgivematchingoutputs.'};(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'prefix_count':len(prefix),'output_count':len(gold),'artifact_count':len(prefix)+len(gold),'paths':len(stats),'max_abs_oracle':max(abs(x) for x in gold if x!=-12345.)}))
