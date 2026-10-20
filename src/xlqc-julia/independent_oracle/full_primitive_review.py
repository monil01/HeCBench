from pathlib import Path
import numpy as np,math,ctypes,json,time,hashlib
P=Path(__file__).resolve().parent;A=Path('/tmp/hecbench-stencil-ports-xlqc-fixtures/example')
lines=(A/'basis.dat').read_text().splitlines();blocks={};at=0
while at<len(lines):
 line=lines[at].strip();at+=1
 if not line or line=='****':continue
 atom=line.split()[0];shells=[]
 while at<len(lines)and lines[at].strip()!='****':
  row=lines[at].split();at+=1
  if not row:continue
  typ=row[0];n=int(row[1]);assert float(row[2])==1.;rows=[[float(x)for x in lines[at+k].split()]for k in range(n)];at+=n;shells.append((typ,rows))
 blocks[atom]=shells
coords=(A/'geom.xyz').read_text().splitlines()[2:];packed=[];mapping=[];cbf=0
for line in coords:
 if not line.strip():continue
 atom,*xyz=line.split();xyz=[float(v)/.5291772086 for v in xyz]
 for typ,rows in blocks[atom]:
  angular=[(0,0,0)]if typ=='S'else[(0,0,0),(1,0,0),(0,1,0),(0,0,1)];assert typ in ['S','SP']
  for angular_index,lmn in enumerate(angular):
   for row in rows:
    alpha=row[0];coef=row[2]if typ=='SP'and angular_index>0 else row[1];L=sum(lmn);den=math.pow(math.pi,1.5)
    for l in lmn:
     for k in range(2*l-1,0,-2):den*=k
    norm=math.sqrt(math.pow(2.,1.5+2*L)*math.pow(alpha,1.5+L)/den);packed.append(xyz+list(lmn)+[alpha,coef*norm]);mapping.append(cbf)
   cbf+=1
p=np.ascontiguousarray(packed,np.float64);assert p.shape==(21,8)and cbf==7;np.save(P/'reconstructed_H2O_pbf.npy',p);np.save(P/'reconstructed_H2O_pbf_to_cbf.npy',np.array(mapping,np.int32));tables={};components={};tm={}
for label in ['original','tight']:
 lib=ctypes.CDLL(str(P/(label+'_eri.so')));fn=lib.review_eri_table;fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];fn.restype=None;arr=np.empty((21,)*4,np.float64);t=time.monotonic();fn(p.ctypes.data,21,arr.ctypes.data);tm[label]=time.monotonic()-t;assert np.all(np.isfinite(arr));tables[label]=arr;np.save(P/(label+'_ERI.npy'),arr)
 b=lib.review_boys;b.argtypes=[ctypes.c_int,ctypes.c_double];b.restype=ctypes.c_double;components[label]=b
xg,wg=np.polynomial.legendre.leggauss(128);tg=(xg+1)/2;wg=wg/2;boys=[]
for m in range(5):
 for x in [0,1e-13,1e-11,1e-8,1e-4,.1,1,3,10,30,80,300]:
  ref=float(np.sum(wg*tg**(2*m)*np.exp(-x*tg*tg)));ref=1/(2*m+1)if x==0 else ref;old=components['original'](m,x);new=components['tight'](m,x);boys.append(dict(m=m,x=x,original=old,tight=new,quadrature128=ref,original_abs_error=abs(old-ref),tight_abs_error=abs(new-ref)))
diff=np.abs(tables['original']-tables['tight']);idx=np.unravel_index(int(np.argmax(diff)),diff.shape);tight=tables['tight'];sym=dict(pair_ij=float(np.max(abs(tight-tight.transpose(1,0,2,3)))),pair_kl=float(np.max(abs(tight-tight.transpose(0,1,3,2)))),pair_exchange=float(np.max(abs(tight-tight.transpose(2,3,0,1)))))
report=dict(status='full_CPU_primitive_uncertainty_comparison_complete',quartets=21**4,all_values_finite=True,max_absolute_original_vs_tight=float(diff.max()),max_error_indices=list(map(int,idx)),max_error_original=float(tables['original'][idx]),max_error_tight=float(tables['tight'][idx]),original_vs_tight_relative_to_peak=float(diff.max()/max(abs(tight).max(),1e-300)),full_ERI_symmetry_errors=sym,Boys_components=boys,max_tight_Boys_vs_independent_quadrature_error=max(z['tight_abs_error']for z in boys),CPU_elapsed_seconds=tm,packing_scope='IndependentPythonreconstructionfromexactrecoveredH2O/STO3Gassets andsourcegeometryconversion/angularorder/normalizationformula; MUST compareagainstactualnativepackedreadback beforeadoptingasstudyinputoracle. NotclaimGPUagreement.',files={str(f):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in [A/'basis.dat',A/'geom.xyz',P/'reconstructed_H2O_pbf.npy',P/'tight_ERI.npy',P/'original_ERI.npy']},GPU_execution=False,source_edits=False);(P/'full_primitive_cpu_review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:report[k]for k in ['quartets','max_absolute_original_vs_tight','max_tight_Boys_vs_independent_quadrature_error','CPU_elapsed_seconds','full_ERI_symmetry_errors']})
