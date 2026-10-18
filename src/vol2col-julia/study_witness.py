"""Independent sparse lowering map and U*U overlap-count witness."""
import sys,json,struct,hashlib
from pathlib import Path
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());prefix=[];expected=[];summary=[]
for sd in (2,1):
 for k in (1,3,5,7,9):
  x=[(i*17+3*k+sd)%101-50 for i in range(630)];prefix.extend(v/16 for v in x);pars=[2,3,5,7,k,k,k,1,1,1,sd,sd,sd,sd,sd,sd,3,5,7,3];prefix.extend(pars);size=(2*k**3+1)*192;col=[0]*size;counts=[0]*210
  # Decode the final flattened column index, independent of native thread loops.
  for q in range(2*k**3*105):
   kernel_channel,spatial=divmod(q,105);ch,rem=divmod(kernel_channel,k**3);kt,rem=divmod(rem,k*k);kh,kw=divmod(rem,k);ot,rem=divmod(spatial,35);oh,ow=divmod(rem,7);t=ot*sd-1+kt*sd;h=oh*sd-1+kh*sd;w=ow*sd-1+kw*sd
   if 0<=t<3 and 0<=h<5 and 0<=w<7:
    idx=((ch*3+t)*5+h)*7+w;col[q]=x[idx];counts[idx]+=1
  vol=[counts[i]*x[i] for i in range(210)]+x[210:];expected.extend(v/16 for v in col);expected.extend(v/16 for v in vol);summary.append({'stride_dilation':sd,'kernel_size':k,'logical_volume_values':210,'physical_volume_values':630,'logical_column_values':2*k**3*105,'physical_column_values':size,'input_voxels_with_positive_coverage':sum(c>0 for c in counts),'max_overlap_count':max(counts),'coverage_histogram':{str(n):counts.count(n)for n in sorted(set(counts))},'composition_not_inverse':any(c!=1 for c in counts)})
allvalues=tuple(prefix+expected);assert len(prefix)==6500 and len(expected)==949020 and len(allvalues)==955520;reports=[]
for model,m in r['models'].items():
 for tr in m['trials']:
  p=Path(tr['output_artifact']['saved_path']);raw=p.read_bytes();v=struct.unpack('<955520f',raw);assert v==allvalues,(model,tr['trial'],[(i,a,b)for i,(a,b)in enumerate(zip(v,allvalues))if a!=b][:5]);reports.append({'model':model,'trial':tr['trial'],'exact_full_columns_volume':True,'all_output_values':949020,'input_control_values':6500,'sha256':hashlib.sha256(raw).hexdigest()})
w={'benchmark':'vol2col','result':str(rp),'captured_trials_checked':len(reports),'method':'Independent column-flat-index sparse map, boundary masks and per-input overlap counts; checks full columns and col2vol(vol2col(x))=coverage*x plus untouched initialized physical tails. Python arbitrary-precision integer numerator arithmetic.','scope':'Backward tested on forward-derived columns as original pipeline, not arbitrary independent column gradients; this is a transpose composition, not inverse or normalization. Explicit column spatial shape retained, not derived standard convolution output shape.','cases':summary,'reports':reports};p=Path('.porting-state/portable_julia/scientific_witnesses_300/vol2col.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
