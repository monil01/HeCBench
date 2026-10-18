"""Independent full-output LU witness using exact rational PA=LU products."""
import json,struct,sys,hashlib
from pathlib import Path
from fractions import Fraction
result=Path(sys.argv[1]).resolve();r=json.loads(result.read_text());reports=[]
for model,m in r['models'].items():
 for tr in m['trials']:
  artifact=Path(tr['output_artifact']['saved_path']);raw=artifact.read_bytes();a=struct.unpack('<'+str(len(raw)//8)+'d',raw);inp=a[:9216];assert a[9216:9221]==(48.,4.,1.,4.,1.);f=a[9221:18437];p=a[18437:18629];info=a[18629:18633];assert info==(0.,0.,1.,0.) and a[18633]==0;worst=Fraction(0);relative=Fraction(0);zero_residual=Fraction(0)
  for b in range(4):
   perm=list(range(48))
   for k in range(48):
    q=int(p[b*48+k])-1;assert k<=q<48;perm[k],perm[q]=perm[q],perm[k]
   for j in range(48):
    for i in range(48):
     x=sum((Fraction(1)if i==k else Fraction(f[b*2304+k*48+i])if i>k else Fraction(0))*(Fraction(f[b*2304+j*48+k])if k<=j else Fraction(0))for k in range(48))
     ref=Fraction(inp[b*2304+j*48+perm[i]]);res=abs(x-ref);worst=max(worst,res)
     if ref:relative=max(relative,res/abs(ref))
     else:zero_residual=max(zero_residual,res)
  assert float(worst)<2e-6 and float(relative)<1e-6
  # A checker that ignores exact-zero reference entries can accept this false answer.
  reports.append({'model':model,'trial':tr['trial'],'artifact_sha256':hashlib.sha256(raw).hexdigest(),'exact_rational_max_component_residual':float(worst),'source_nonzero_relative_bound':1e-6,'max_nonzero_relative_error':float(relative),'max_zero_reference_abs_residual':float(zero_residual),'all_factor_values':9216,'all_pivot_values':192,'all_info_values':4,'status':0})
out={'benchmark':'ludb','result':str(result),'captured_trials_checked':len(reports),'method':'Exact rational reconstruction of P*A and L*U from complete actual Float32 input/factors; validates all pivots and singular info. No shared GPU checker arithmetic.','legacy_checker_blindspot':{'reference':[0,1],'wrong':[9,1],'native_skips_zero_reference':True,'scope':'Static minimal checker limitation, not observed factorization defect.'},'scope':'Named dimension48 batch4; singular zero and pivoted/diagonal/dense cases. Not arbitrary conditioning, backward stability survey, or framework-authored LU.','reports':reports};dest=Path('.porting-state/portable_julia/scientific_witnesses_300/ludb.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
