#!/usr/bin/env python3
"""Independent CPU dense LongDouble CNN operator audit. No CUDA/library solver calls.
Requires numpy; e.g. recorded environment PYTHONPATH=/tmp/stencil_ports_numpy_226.
Consumes complete actual staged GPU outputs, not a shared generated solver result.
"""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
D=np.longdouble
U=D(2)**-24
FIELDS=('output','preact','bias','weight','d_output','d_preact','d_weight')
SHAPES=((0,0,784),(25,6,3456),(16,1,216),(216,10,10))
SIZES=[(o,o,n,m*n,o,o,m*n) for m,n,o in SHAPES]
LENGTH=sum(n for l,s in enumerate(SIZES) for k,n in enumerate(s) if n and not(l==0 and k in(4,5)))
def gamma(k):return D(k)*U/(1-D(k)*U)
def sigmoid(x):return 1/(1+np.exp(-x))
def read_records(path):
 raw=path.read_bytes(); recsize=16+LENGTH*4
 if len(raw)%recsize:raise ValueError('truncated stage record')
 for offset in range(0,len(raw),recsize):
  h=struct.unpack_from('<4i',raw,offset); a=np.frombuffer(raw,dtype='<f4',count=LENGTH,offset=offset+16);ls=[];j=0
  for l,s in enumerate(SIZES):
   layer={}
   for k,n in enumerate(s):
    if n and not(l==0 and k in(4,5)):layer[FIELDS[k]]=a[j:j+n].astype(D);j+=n
   ls.append(layer)
  assert j==LENGTH
  yield h,ls
class Auditor:
 def __init__(self):self.count=0;self.max_abs=0.;self.max_fraction=0.;self.operators={};self.failures=[]
 def check(self,name,actual,ref,bound):
  a=np.asarray(actual,dtype=D);r=np.asarray(ref,dtype=D);b=np.broadcast_to(np.asarray(bound,dtype=D),a.shape)
  if a.shape!=r.shape:raise ValueError((name,a.shape,r.shape))
  e=np.abs(a-r);frac=e/np.maximum(b,D('1e-4900'));ok=np.isfinite(a)&np.isfinite(r)&(e<=b)
  self.count+=a.size;self.max_abs=max(self.max_abs,float(np.max(e,initial=0)));self.max_fraction=max(self.max_fraction,float(np.max(frac,initial=0)));self.operators[name]=self.operators.get(name,0)+a.size
  if not np.all(ok):
   ids=np.flatnonzero(~ok);i=int(ids[0]);self.failures.append({'operator':name,'failures':len(ids),'first_index':i,'actual':float(a.flat[i]),'reference':float(r.flat[i]),'bound':float(b.flat[i]),'max_fraction':float(np.max(frac))})
 def reduction(self,name,a,terms,bias=0):
  r=terms.sum(axis=0,dtype=D)+bias; mag=np.abs(terms).sum(axis=0,dtype=D)
  self.check(name,a,r,gamma(2*terms.shape[0]+8)*mag+gamma(8)*np.abs(bias)+D(1e-35))
def forward(A,ls,label):
 image=ls[0]['output'].reshape(28,28)
 c,s,f=ls[1:];cw=c['weight'].reshape(6,5,5);co=c['output'].reshape(6,24,24);cp=c['preact'].reshape(6,24,24)
 terms=np.array([cw[:,kr,kc,None,None]*image[kr:kr+24,kc:kc+24][None,:,:] for kr in range(5) for kc in range(5)],dtype=D)
 A.reduction('C1forward_convolution',cp,terms,c['bias'][:,None,None]);A.check('C1sigmoid',co,sigmoid(cp),12*U)
 sw=s['weight'].reshape(4,4);sp=s['preact'].reshape(6,6,6);so=s['output'].reshape(6,6,6)
 terms=np.array([sw[kr,kc]*co[:,kr:kr+24:4,kc:kc+24:4] for kr in range(4) for kc in range(4)],dtype=D)
 A.reduction('S1learned_filter',sp,terms,s['bias'][0]);A.check('S1sigmoid',so,sigmoid(sp),12*U)
 fw=f['weight'].reshape(10,216);terms=(fw*so.reshape(1,216)).T
 A.reduction('Fforward',f['preact'],terms,f['bias']);A.check('Fsigmoid',f['output'],sigmoid(f['preact']),12*U)
 if label is not None:
  expected=-f['output'].copy();expected[label]+=1
  A.check('negative_BCE_top_delta',f['d_preact'],expected,2*U)
def backward(A,old,new):
 c,s,f=old[1:];nc,ns,nf=new[1:];fd=old[3]['d_preact'];so=s['output'];co=c['output'].reshape(6,24,24);image=old[0]['output'].reshape(28,28)
 A.check('Fweight_gradient',nf['d_weight'].reshape(10,216),fd[:,None]*so[None,:],gamma(2)*np.abs(fd[:,None]*so[None,:])+D(1e-35))
 terms=(f['weight'].reshape(10,216)*fd[:,None]);A.reduction('S1reverse_dense',ns['d_output'],terms)
 sr=sigmoid(s['preact']);sd=ns['d_output']*sr*(1-sr)
 A.check('S1chain_delta',ns['d_preact'],sd,32*U*np.maximum(1,np.abs(sd)))
 sd=ns['d_preact'].reshape(6,6,6)
 for kr in range(4):
  for kc in range(4):
   terms=(sd*co[:,kr:kr+24:4,kc:kc+24:4]).reshape(-1,1)
   A.reduction('S1shared_weight_gradient',np.array([ns['d_weight'][kr*4+kc]]),terms)
 cdo=np.zeros((6,24,24),dtype=D);sw=s['weight'].reshape(4,4)
 for kr in range(4):
  for kc in range(4):cdo[:,kr:kr+24:4,kc:kc+24:4]=sw[kr,kc]*sd
 A.check('C1reverse_strided',nc['d_output'].reshape(6,24,24),cdo,gamma(2)*np.abs(cdo)+D(1e-35))
 cr=sigmoid(c['preact']);cd=nc['d_output']*cr*(1-cr)
 A.check('C1chain_delta',nc['d_preact'],cd,32*U*np.maximum(1,np.abs(cd)))
 cd=nc['d_preact'].reshape(6,24,24)
 for ch in range(6):
  for kr in range(5):
   for kc in range(5):
    terms=(cd[ch]*image[kr:kr+24,kc:kc+24]/D(576)).reshape(-1,1)
    A.reduction('C1averaged_weight_gradient',np.array([nc['d_weight'][ch*25+kr*5+kc]]),terms)
 for l,nl,div in ((f,nf,1),(s,ns,216),(c,nc,576)):
  if div==1:inc=D(np.float32(.1))*fd;ref=l['bias']+inc;bound=gamma(4)*(np.abs(l['bias'])+np.abs(inc))
  else:
   rows=nl['d_preact'].reshape(-1,div) if div==576 else nl['d_preact'].reshape(1,216)
   inc=D(np.float32(.1))*rows/D(div);ref=l['bias']+inc.sum(axis=1,dtype=D);bound=gamma(2*div+8)*(np.abs(l['bias'])+np.abs(inc).sum(axis=1,dtype=D))
  A.check('parameter_bias_update_'+str(div),nl['bias'],ref,bound+D(1e-35))
  inc=D(np.float32(.1))*nl['d_weight'];ref=l['weight']+inc
  A.check('parameter_weight_update_'+str(div),nl['weight'],ref,gamma(4)*(np.abs(l['weight'])+np.abs(inc))+D(1e-35))
def physical(path):
 raw=path.read_bytes();offset=0;guardbytes=0;ids=[]
 for expected_id,n in enumerate(n for sizes in SIZES for n in sizes):
  i,size=struct.unpack_from('<QQ',raw,offset);offset+=16
  if(i,size)!=(expected_id,n*4):raise ValueError('physical allocation layout mismatch')
  block=raw[offset:offset+size+512];offset+=size+512
  if len(block)!=size+512 or block[:256]!=bytes([0xa7])*256 or block[-256:]!=bytes([0xa7])*256:raise ValueError('physical guard mismatch')
  guardbytes+=512;ids.append(i)
 if offset!=len(raw):raise ValueError('trailing physical bytes')
 return {'raw_bytes':len(raw),'actual_guard_bytes':guardbytes,'allocation_ids':ids,'sha256':hashlib.sha256(raw).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.folder);A=Auditor();receipts=[]
 for trial in range(1,6):
  science=folder/f'{a.model}_trial{trial}.science.bin';prev=None;counts={0:0,1:0,2:0};labels=[]
  for h,ls in read_records(science):
   kind,step,label,pred=h;counts[kind]+=1
   if kind==0:
    if prev is not None:raise ValueError('missing previous backprop')
    forward(A,ls,label);prev=(h,ls)
   elif kind==1:
    if prev is None or prev[0][1:3]!=(step,label):raise ValueError('backprop joins mismatch')
    backward(A,prev[1],ls);prev=None
   elif kind==2:
    forward(A,ls,None);expected=int(np.argmax(ls[3]['output']));A.check('classification_first_argmax',np.array([pred],dtype=D),np.array([expected],dtype=D),0);labels.append((label,pred))
  if counts!={0:200,1:200,2:20} or prev is not None:raise ValueError(('incomplete source pipeline',counts))
  receipts.append({'trial':trial,'stage_counts':counts,'science_sha256':hashlib.sha256(science.read_bytes()).hexdigest(),'science_bytes':science.stat().st_size,'physical':physical(folder/f'{a.model}_trial{trial}.physical.bin'),'test_label_prediction_pairs':labels})
 out={'benchmark':'mnist','model':a.model,'correct':not A.failures,'independent_comparisons':A.count,'max_abs_error':A.max_abs,'max_declared_bound_fraction':A.max_fraction,'operators':A.operators,'failures':A.failures,'LongDouble_precision_bits':int(np.finfo(D).nmant+1),'trials':receipts,'scope':'Local source CNN operators given actual staged GPU inputs/weights/activations; no full-dataset/convergence/classification-quality certificate.'}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('correct','independent_comparisons','max_abs_error','max_declared_bound_fraction')}));return 0 if out['correct'] else 1
if __name__=='__main__':raise SystemExit(main())
