#!/usr/bin/env python3
"""CPU-only independent chain/finite-difference witness of source gradient policy."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
D=np.longdouble

def evaluate(image,cw,cb,sw,sb,fw,fb,label,gradient=False):
 cp=np.broadcast_to(cb[:,None,None],(6,24,24)).copy()
 for kr in range(5):
  for kc in range(5):cp+=cw[:,kr,kc,None,None]*image[kr:kr+24,kc:kc+24]
 co=1/(1+np.exp(-cp));sp=np.full((6,6,6),sb[0],dtype=D)
 for kr in range(4):
  for kc in range(4):sp+=sw[kr,kc]*co[:,kr:kr+24:4,kc:kc+24:4]
 so=1/(1+np.exp(-sp));z=fw@so.reshape(216)+fb
 # Stable BCE over ten independent sigmoid logits; not softmax.
 y=np.zeros(10,dtype=D);y[label]=1;loss=np.sum(np.maximum(z,0)-y*z+np.log1p(np.exp(-np.abs(z))),dtype=D)
 if not gradient:return loss
 fo=1/(1+np.exp(-z));fd=y-fo
 fg=fd[:,None]*so.reshape(1,216);sd=(fw.T@fd).reshape(6,6,6)*so*(1-so)
 sg=np.empty((4,4),dtype=D);cdo=np.empty((6,24,24),dtype=D)
 for kr in range(4):
  for kc in range(4):sg[kr,kc]=np.sum(sd*co[:,kr:kr+24:4,kc:kc+24:4],dtype=D);cdo[:,kr:kr+24:4,kc:kc+24:4]=sw[kr,kc]*sd
 cd=cdo*co*(1-co);cg=np.empty((6,5,5),dtype=D)
 for kr in range(5):
  for kc in range(5):cg[:,kr,kc]=np.sum(cd*image[kr:kr+24,kc:kc+24],axis=(1,2),dtype=D)
 return loss,[cg,cd.sum(axis=(1,2),dtype=D),sg,np.array([sd.sum(dtype=D)]),fg,fd]
def main():
 p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args();raw=Path(a.initial).read_bytes();v=np.frombuffer(raw,dtype='<f4').astype(D);assert len(v)==2343
 cb=v[:6].copy();cw=v[6:156].reshape(6,5,5).copy();sb=v[156:157].copy();sw=v[157:173].reshape(4,4).copy();fb=v[173:183].copy();fw=v[183:].reshape(10,216).copy();case=Path(a.case);images=(case/'train-images.idx3-ubyte').read_bytes();labels=(case/'train-labels.idx1-ubyte').read_bytes();image=np.frombuffer(images,dtype='u1',offset=16,count=784).astype(D).reshape(28,28)/255
 # Actual host loader first rounds /255 to Float64 then upload FP32.
 image=image.astype(np.float64).astype(np.float32).astype(D);label=labels[8]
 loss,g=evaluate(image,cw,cb,sw,sb,fw,fb,label,True);arrays=[cw,cb,sw,sb,fw,fb];indices=[(2,3,1),(4,),(1,2),(0,),(7,117),(5,)];rows=[];h=D('0.00001')
 for k,(array,idx) in enumerate(zip(arrays,indices)):
  old=array[idx];array[idx]=old+h;lp=evaluate(image,cw,cb,sw,sb,fw,fb,label);array[idx]=old-h;lm=evaluate(image,cw,cb,sw,sb,fw,fb,label);array[idx]=old
  numerical=(lp-lm)/(2*h);analytic=-g[k][idx];error=abs(numerical-analytic);bound=D('0.0000001')*max(D(1),abs(analytic))
  rows.append({'parameter_group':('Cweight','Cbias','Sweight','Sbias','Fweight','Fbias')[k],'index':idx,'negative_BCE_gradient':float(g[k][idx]),'centered_derivative':float(numerical),'absolute_error':float(error),'named_finite_difference_budget':float(bound),'correct':bool(error<=bound),'source_update_preconditioner':float(1/D(576)) if k<2 else float(1/D(216)) if k==3 else 1.0})
 out={'benchmark':'mnist','correct':all(r['correct'] for r in rows),'oracle':'IndependentLongDouble fullCNN BCEchain+centeredfinite-difference, actualsourceinitialparams+firstactualMNISTimage. CPU-only mathwitness, notGPUcandidate.','loss':float(loss),'parameters_sha256':hashlib.sha256(raw).hexdigest(),'LongDouble_precision_bits':int(np.finfo(D).nmant+1),'step':float(h),'rows':rows,'scope':'MixedaveragingcorrespondstopositiveparameterpreconditioningofnegativeBCEgradient. SourceEuclideandiagnostic/earlystop notclaimedBCEobjective, noMSE-gradientdefectclaim.'};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'correct':out['correct'],'rows':len(rows),'max_abs_error':max(r['absolute_error'] for r in rows)}));return 0 if out['correct'] else 1
if __name__=='__main__':raise SystemExit(main())
