"""Independent CPU operator/threshold controls over actual complete GPU warm traces.
No audit/capture runs in the timed primary region; overwritten timed states are not reconstructed.
"""
from pathlib import Path
import json,os,sys,hashlib
import numpy as np
P=Path(__file__).resolve().parent;D=Path(sys.argv[1]);plan=json.loads((P/'study_plan.json').read_text());raw=np.fromfile(P/'input_u32.bin',dtype='<u4');rf=raw.view('<f4');dense=np.fromfile(P/'dense_oracle_f64.bin',dtype='<f8');stats=[]
def arr(c,key,shape,typ='f4'):
 off=c[key];n=int(np.prod(shape));return raw[off:off+n].view('<'+typ).reshape(shape)
def mat(c,l):
 e=c['layers'][l];rp=arr(e,'rp',(c['H']+1,),'i4');col=arr(e,'col',(e['nnz'],),'i4');v=arr(e,'val',(e['nnz'],));w=np.zeros((c['H'],c['H']),dtype='f8');np.add.at(w,(np.repeat(np.arange(c['H']),np.diff(rp)),col),v);return w
mx=0.;budget=0.;checked=0;counts={}
def close(a,b,exact=False):
 global mx,budget,checked
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
 assert np.isfinite(a).all() and np.isfinite(b).all()
 diff=np.abs(a.astype('f8')-b);err=float(diff.max(initial=0));mx=max(mx,err);checked+=a.size
 if exact:assert np.array_equal(a,b),("categorical/unchanged mismatch",err)
 else:
  allowed=1e-4+1e-4*np.abs(b);ratio=float(np.max(diff/allowed,initial=0));budget=max(budget,ratio);assert ratio<=1,("numeric budget",ratio,err)
def readstate(f,c):
 n=2*c['M']*c['H']+c['M']*10;nf=np.fromfile(f,dtype='<f4',count=n);ni=np.fromfile(f,dtype='<i4',count=128+3*c['M']+1);assert len(nf)==n and len(ni)==128+3*c['M']+1
 m,h=c['M'],c['H'];return [nf[:m*h].reshape(m,h),nf[m*h:2*m*h].reshape(m,h),nf[2*m*h:].reshape(m,10),ni[:128],ni[128:128+m],ni[128+m:128+2*m],ni[128+2*m:128+3*m],ni[-1:]]
for c in plan['cases']:
 M,H,N=c['M'],c['H'],c['N'];weights=[mat(c,l)for l in range(c['L'])];bias=[arr(e,'bias',(H,))for e in c['layers']];trace=D/f"trace_case{c['case']}.bin";events=[]
 with trace.open('rb')as f:
  while True:
   hd=np.fromfile(f,dtype='<i4',count=5)
   if not len(hd):break
   assert len(hd)==5;op,ci,l,C,grid=map(int,hd);assert ci==c['case'];before=readstate(f,c);after=readstate(f,c);want=[x.copy()for x in before];h=c['T']%2
   if op==0:want[0]=np.clip(arr(c,'x',(M,N)).astype('f8')@arr(c,'win',(N,H)).astype('f8')+arr(c,'bin',(H,)),0,1)
   elif op==1:want[(l+1)%2]=np.clip(before[l%2].astype('f8')@weights[l]+bias[l],0,1)
   elif op==2:want[2]=before[h].astype('f8')@arr(c,'wout',(H,10)).astype('f8')+arr(c,'bout',(10,))
   elif op==3:want[7][0]=np.count_nonzero(before[2].argmax(axis=1)==arr(c,'label',(M,),'i4'))
   elif op==4:
    seeds=(np.arange(128)*M//128).astype('i4');source=before[h]
    for i in range(128):
     if seeds[i]!=-1:
      delta=np.abs(source[(np.arange(128)*M//128)]-source[seeds[i]])
      n=np.count_nonzero(delta>np.float32(.03),axis=1)
      for j in range(128):
       if j!=i and seeds[j]!=-1 and n[j]<np.float32(H)*np.float32(.03):seeds[j]=-1
    want[3]=seeds
   elif op==5:
    source=before[h];seed=before[3][:C];assert C>0 and np.all(seed>=0) and np.all(seed<M)
    for row in range(M):
     if before[4][row]==-1:want[5][row]=1;continue
     n=np.count_nonzero(np.abs(source[seed]-source[row])>np.float32(.04),axis=1);best=int(seed[n.argmin()]);want[4][row]=best;delta=source[row]-source[best];v=np.where(np.abs(delta)>np.float32(.04),delta,np.float32(0));want[h][row]=v;want[5][row]=np.any(v>0)
   elif op==6:
    rows=before[6][:C];assert len(np.unique(rows))==C and np.all(rows>=0) and np.all(rows<M);want[1-h][rows]=before[h][rows].astype('f8')@weights[l]
   elif op==7:
    for row in before[6][:C]:
     if before[4][row]==-1:want[h][row]=np.clip(before[1-h][row]+bias[l],0,1);want[5][row]=1
     else:
      wy=before[1-h][before[4][row]];delta=before[1-h][row];base=np.clip(wy+bias[l],0,1);td=np.clip((wy+bias[l])+delta,0,1)-base;v=np.where(np.abs(td).astype('f8')>.05,td,np.float32(0));want[h][row]=v;want[5][row]=np.any(v!=0)
   elif op==8:
    for row in range(M):
     if before[4][row]!=-1:want[h][row]=before[h][row]+before[h][before[4][row]]
   else:raise AssertionError(op)
   for i,(a,b)in enumerate(zip(after,want)):close(a,b,exact=i>=3)
   counts[str(op)]=counts.get(str(op),0)+1;events.append(dict(op=op,layer=l,active_or_seed_count=C,grid=grid))
 stats.append(dict(case=c['name'],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),trace_bytes=trace.stat().st_size,events=events))
# Independent final-logit and exact classification/metadata contract for every timed full surviving state.
offset=plan['input_words'];quality=[];categorical=[]
for trial in range(1,6):
 f=D/f'trial{trial}.bin';v=np.fromfile(f,dtype='<f8');assert np.array_equal(v[:offset],raw.astype('f8'));pos=offset;qs=[];cats=[]
 for c in plan['cases']:
  M,H=c['M'],c['H'];h=[]
  for _ in range(2):h.append(v[pos:pos+M*H].reshape(M,H));pos+=M*H
  out=v[pos:pos+M*10].reshape(M,10);pos+=M*10
  ys=v[pos:pos+128];pos+=128;lut=v[pos:pos+M];pos+=M;ne=v[pos:pos+M];pos+=M;rows=v[pos:pos+M];pos+=M;cnt=v[pos];pos+=1;pred=v[pos:pos+M];pos+=M
  # Guard allocation inventory: x, optionalwin/bin,h0/h1/out/wo/bo,ys/lut/rows/labels/cnt/ne, fourperhiddenlayer.
  na=1+(0 if c['cifar']else 2)+5+6+4*c['L'];guards=v[pos:pos+16*na];pos+=16*na;assert len(guards)==16*na and np.isin(guards,[1,91]).all()
  expected=h[c['T']%2]@arr(c,'wout',(H,10)).astype('f8')+arr(c,'bout',(10,));close(out,expected);close(pred,out.argmax(axis=1),True);close(np.array([cnt]),np.array([np.count_nonzero(pred==arr(c,'label',(M,),'i4'))]),True)
  assert np.isin(ne,[0,1]).all() and np.all((lut>=-1)&(lut<M)) and np.all((rows>=0)&(rows<M))
  cats.extend([*ys,*lut,*ne,*rows,cnt,*pred]);dd=dense[c['dense_oracle_offset']:c['dense_oracle_offset']+M*10].reshape(M,10)
  qs.append(dict(case=c['name'],correct=int(cnt),examples=M,real_dataset_accuracy=not c['cifar'],dense_independent_correct=c['dense_reference_correct_labels'],dense_argmax_disagreements=int(np.count_nonzero(pred!=dd.argmax(axis=1))),dense_logit_max_abs=float(np.abs(out-dd).max())))
 assert pos==len(v),(pos,len(v));categorical.append(np.array(cats,dtype='<i4'));quality.append(qs)
 # Exact branch/classifier outputs must not be hidden by a float budget.
 if trial>1:assert np.array_equal(categorical[0],categorical[-1]),'within-model categorical outcomes vary across trials'
report=dict(correct=True,checked_operator_values=checked,max_abs_operator_error=mx,max_normalized_operator_budget=budget,source_kernel_entrypoints=9,trace_entrypoint_counts=counts,actual_untimed_traces=stats,timed_capture_scope='all surviving states, overwritten intermediates not claimed',categorical_trial_sha256=[hashlib.sha256(a.tobytes()).hexdigest()for a in categorical],timed_quality_vs_dense_independent=quality,quality_is_not_translation_acceptance=True,CPU_oracle='independentNumPyFloat64linearalgebra on actualGPUstageinputs; sourceFloat32thresholdandclamp controls; greedyseed/argmin/argmax exact')
(D/'independent_cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('SNICIT_CPU_AUDIT',json.dumps({k:report[k]for k in ['correct','checked_operator_values','max_abs_operator_error','max_normalized_operator_budget','trace_entrypoint_counts']}),flush=True)
