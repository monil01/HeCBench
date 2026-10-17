"""Independent reconstruction and tie/stopping-scope checks on saved full captures.
No GPU work; use the actual serialized inputs and actual measured outputs.
"""
import argparse,array,ctypes,json,hashlib
from pathlib import Path
libm=ctypes.CDLL('libm.so.6');fmaf=libm.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
f32=lambda x:ctypes.c_float(x).value
WIDTHS=(65,257);PREFIX=140658

def read(path):
 a=array.array('d');a.frombytes(Path(path).read_bytes());return a

def oracle(dirty,psf,w,gain,iters,hierarchical):
 res=list(dirty);model=[0.]*len(res);trace=[]
 for _ in range(iters):
  maximum=max(abs(v)for v in res)
  candidates=(i for i,v in enumerate(res)if abs(v)==maximum)
  pos=min(candidates,key=(lambda i:(i%32768,i//32768))if hierarchical else None);val=res[pos];trace.extend([pos,val]);factor=f32(gain*val);model[pos]=f32(model[pos]+factor);sx,sy=pos%w,pos//w
  for y in range(max(0,sy-3),min(w-1,sy+3)+1):
   for x in range(max(0,sx-3),min(w-1,sx+3)+1):
    k=y*w+x;res[k]=fmaf(-factor,psf[(y-sy+3)*7+x-sx+3],res[k])
 return model,res,trace

def audit(result):
 d=json.loads(Path(result).read_text());reports=[]
 for model,info in d['models'].items():
  for trial in info.get('trials',[]):
   art=trial.get('output_artifact',{});path=art.get('saved_path')
   if not path:continue
   a=read(path);off=0;inputs=[]
   for w in WIDTHS:
    n=w*w;dirty=a[off:off+n];off+=n;psf=a[off:off+49];off+=49;initial=a[off:off+n];off+=n;control=a[off:off+6];off+=6
    assert all(v==0 for v in initial);assert control[0]==w and control[1]==7 and control[2]==19;inputs.append((w,dirty,psf,control))
   assert off==PREFIX
   cases=[]
   for w,dirty,psf,control in inputs:
    n=w*w;actualmodel=a[off:off+n];off+=n;residual=a[off:off+n];off+=n;psfpeak=a[off:off+2];off+=2;trace=a[off:off+38];off+=38
    wantmodel,wantres,wanttrace=oracle(dirty,psf,w,control[3],19,True)
    err=max(max(abs(x-y)for x,y in zip(actualmodel,wantmodel)),max(abs(x-y)for x,y in zip(residual,wantres)),max(abs(x-y)for x,y in zip(trace,wanttrace)))
    assert err<=.001;assert trace[::2]==array.array('d',wanttrace[::2]);assert min(abs(v)for v in trace[1::2])>control[4];assert list(psfpeak)==[24,1]
    # Distinct conservation-like identity: dirty = residual + shifted PSF * model.
    reconstructed=list(residual)
    for pos,amplitude in enumerate(actualmodel):
     if amplitude==0:continue
     sx,sy=pos%w,pos//w
     for y in range(max(0,sy-3),min(w-1,sy+3)+1):
      for x in range(max(0,sx-3),min(w-1,sx+3)+1):reconstructed[y*w+x]+=amplitude*psf[(y-sy+3)*7+x-sx+3]
    identityerror=max(abs(x-y)for x,y in zip(reconstructed,dirty));assert identityerror<=.001
    report={'width':w,'full_state_recurrence_max_abs_error':err,'dirty_equals_residual_plus_shifted_psf_model_max_abs_error':identityerror,'minimum_measured_peak_absolute':min(abs(v)for v in trace[1::2]),'threshold':control[4],'threshold_stopping_inactive':True,'exact_peak_positions':list(trace[::2]),'psf_peak':list(psfpeak)}
    if w==257:
     othermodel,otherres,othertrace=oracle(dirty,psf,w,control[3],19,False);report.update(global_first_alternative_first_peak=othertrace[0],native_priority_first_peak=trace[0],finite_19iteration_model_difference_global_first=max(abs(x-y)for x,y in zip(actualmodel,othermodel)))
     assert trace[0]==32769 and othertrace[0]==300
    cases.append(report)
   assert off==len(a)
   reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'cases':cases})
 return {'benchmark':'hogbom','source_unchanged_centered_scope':True,'result':str(Path(result).resolve()),'captured_trials_checked':len(reports),'scope':'CenteredunitpeakPSF,19above-thresholditerations; finiteiterationaccuracy notconvergence; noarbitraryPSForoffcenterCLIclaim.','offcenter_index_witness':{'psf_width':5,'peak_x':1,'peak_y':3,'first_overlap_row':'ry-3','native_diffy':'ry-1','computed_psf_y':-2,'classification':'static source-specific scientific-contract limitation, not observed failure for measured centered inputs','repair_backlog':'Independently justified offcenter dy correction configuration, separate from untouched native primary.'},'reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)
