#!/usr/bin/env python3
"""Independent full-trajectory FFT/box-integral mathematical operator oracle."""
from pathlib import Path
import numpy as np,json,struct,hashlib,sys
P=Path(__file__).resolve().parent;R=P.parent.parent;schema=json.load(open(P/'params_schema.json'))['fields'];vals=struct.unpack('<'+''.join('i'if f['type']=='int32'else'f'for f in schema),(P/'common.bin').read_bytes());c={f['name']:v for f,v in zip(schema,vals)}
rows=np.r_[np.fromfile(P/'endoRow_i32.bin',dtype='<i4'),np.fromfile(P/'epiRow_i32.bin',dtype='<i4')];cols=np.r_[np.fromfile(P/'endoCol_i32.bin',dtype='<i4'),np.fromfile(P/'epiCol_i32.bin',dtype='<i4')];frames=[np.fromfile(P/f'frame{k}_f32.bin',dtype='<f4').reshape((c['frame_rows'],c['frame_cols']),order='F').astype(np.float64)for k in range(11)];T=[frames[0][r-26:r+25,col-26:col+25].copy()for r,col in zip(rows,cols)];historyR=[rows.copy()];historyC=[cols.copy()];previousR=rows.copy();previousC=cols.copy();maps=[];covs=[];sums=[];bounds=[];templates=[];peaks=[]
(P/'oracle_template_initial_f32.bin').write_bytes(np.stack(T,axis=2).astype('<f4').ravel(order='F').tobytes())

def conv(a,b):return np.fft.irfftn(np.fft.rfftn(a,s=(131,131),axes=(0,1))*np.fft.rfftn(b,s=(131,131),axes=(0,1)),s=(131,131),axes=(0,1))
def box(a):
 p=np.pad(a,((50,50),(50,50)));cs=np.pad(p,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
 return cs[51:,51:]-cs[:-51,51:]-cs[51:,:-51]+cs[:-51,:-51]
for frame in range(1,11):
 nn=[];vv=[];ss=[];bb=[];newR=[];newC=[]
 for point,(row,col,t)in enumerate(zip(rows,cols,T)):
  x=frames[frame][row-41:row+40,col-41:col+40];assert x.shape==(81,81);sm=box(x);sq=box(x*x);raw=conv(x,t[::-1,::-1]);n=t.size;st=t.sum();st2=(t*t).sum();den=np.sqrt((n-1)*(st2/n-(st/n)**2));cov=raw-sm*st/n;ncc=cov/(np.sqrt(np.maximum(0,sq-sm*sm/n))*den);assert np.all(np.isfinite(ncc))
  centerR=65+previousR[point]-row;centerC=65+previousC[point]-col;mask=np.zeros((131,131));mask[max(0,centerR-5):min(131,centerR+5),max(0,centerC-5):min(131,centerC+5)]=1;masked=np.abs(ncc*mask);flat=masked.ravel(order='F');position=int(np.argmax(flat));rr=position%131;cc=position//131;newR.append(row+rr-65);newC.append(col+cc-65);two=np.partition(flat,-2)[-2:];peaks.append({'frame':frame,'point':point,'best_abs_score':float(flat[position]),'best_second_gap':float(max(two)-min(two))})
  # Conservative standardgamma_n positive-dot summation bound plus subtraction/mean-termrounding.
  unit=2.0**-24;gamma=(2601*unit)/(1-2601*unit);error=gamma*np.maximum(raw,0)+16*unit*(np.abs(raw)+np.abs(sm*st/n))+1e-4
  nn.append(ncc);vv.append(cov);ss.append(sm);bb.append(error)
 maps.append(np.stack(nn,axis=2));covs.append(np.stack(vv,axis=2));sums.append(np.stack(ss,axis=2));bounds.append(np.stack(bb,axis=2));previousR=np.asarray(newR,dtype=np.int32);previousC=np.asarray(newC,dtype=np.int32);historyR.append(previousR.copy());historyC.append(previousC.copy())
 if frame%10==0:
  rows=previousR.copy();cols=previousC.copy();alpha=np.float32(c['alpha']);other=np.float32(1-alpha)
  for p,(row,col)in enumerate(zip(rows,cols)):
   patch=frames[frame][row-26:row+25,col-26:col+25];T[p]=(float(alpha)*T[p]+float(other)*patch).astype(np.float32).astype(np.float64)
for name,data in [('oracle_NCC_f64.bin',np.stack(maps,axis=3)),('oracle_covariance_f64.bin',np.stack(covs,axis=3)),('oracle_boxsum_f64.bin',np.stack(sums,axis=3)),('oracle_covariance_budget_f64.bin',np.stack(bounds,axis=3)),('oracle_template_final_f64.bin',np.stack(T,axis=2)),('oracle_track_rows_i32.bin',np.stack(historyR,axis=1)),('oracle_track_cols_i32.bin',np.stack(historyC,axis=1))]:
 a=np.asarray(data,dtype='<i4'if'i32'in name else'<f8');(P/name).write_bytes(a.ravel(order='F').tobytes())
report={'benchmark':'heartwall','oracle':'IndependentFloat64FFTconvolution/directrectangularintegrals, full51points×10trackingupdates+frame0templateinitialization; sourceGPUdirectfold/cumsums nottranscribed.','parameters':{'tSize':25,'sSize':40,'maxMove':10,'alpha':c['alpha'],'points':51,'frames':11},'source_method_policies':['denomT sqrt((n-1)*population_variance) ratherthancanonicalNCCdenominator','argmaxabsoluteNCC, firstcolumnmajor tie; maxMove10window[-5,+4] relativepreviouscoordinate','templatebaseupdate/blend atframe10'],'predeclared_before_primary_budgets':{'independent_NCC_atol':1e-3,'independent_boxsum_atol':0,'covariance':'Per-elementgamma2601 positive-dotbound+16u subtractionterm+1e-4; encodedfullbudgetarray','tracking_integer_coordinates':0,'final_template_atol':3e-5,'dual_atol':2e-3,'dual_rtol':2e-6,'all_serialized_guards_exact':True},'budget_basis':'NCCengineeringceilinginformedbyreadinessframe1diagnosticmax.000300 plusstrictcategoricalargmax; covdotroundoffstandardconservativegamma_n. No formalclinicalaccuracy/convergencebudget. Declaredbeforeanyfive-trialprimaryorJuliaexecution.','minimum_independent_peak_gap':min(p['best_second_gap']for p in peaks),'peaks':peaks,'numpy':np.__version__,'python':sys.version,'raw_source_GPU_trace_used_as_oracle':False,'clinical_accuracy_claim':False}
(P/'scientific_oracle_plan.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='peaks'})
