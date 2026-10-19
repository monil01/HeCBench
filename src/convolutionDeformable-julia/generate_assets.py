#!/usr/bin/env python3
"""CPU shape-function/adjoint oracle, independent of the source CUDA thread/pointer code."""
from pathlib import Path
import math,struct,array,json,hashlib,sys
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def geometry():
 return [dict(id=0,sample=0,C=4,G=2,H=5,W=7,KH=3,KW=3,PH=1,PW=1,SH=1,SW=1,DH=1,DW=1),dict(id=1,sample=0,C=4,G=2,H=7,W=9,KH=3,KW=3,PH=1,PW=1,SH=2,SW=1,DH=2,DW=1),dict(id=2,sample=1,C=4,G=2,H=7,W=9,KH=3,KW=3,PH=1,PW=1,SH=2,SW=1,DH=2,DW=1),dict(id=3,sample=0,C=6,G=3,H=6,W=8,KH=2,KW=3,PH=0,PW=0,SH=1,SW=2,DH=1,DW=2)]
def dims(s):
 s=dict(s);s['OH']=(s['H']+2*s['PH']-(s['DH']*(s['KH']-1)+1))//s['SH']+1;s['OW']=(s['W']+2*s['PW']-(s['DW']*(s['KW']-1)+1))//s['SW']+1
 return s
FIELDS=['id','sample','C','G','H','W','KH','KW','PH','PW','SH','SW','DH','DW','OH','OW']
def data(s):
 C,G,H,W,KH,KW,OH,OW=[s[k]for k in ['C','G','H','W','KH','KW','OH','OW']];S=OH*OW;K=KH*KW;case=s['id']
 image=[((i*7+case*3)%23-11)/8 for i in range(C*H*W)];offset=[];mask=[]
 for group in range(G):
  for k in range(K):
   for d in range(2):
    offset +=[(.25 if d==0 else .375)*(1 if(i+k+group+case)%2 else -1)for i in range(S)]
   mask +=[.25*(1+(i+2*k+group+case)%3)for i in range(S)]
 # Mathematically valid columns-gradient W^T grad_output; outside timedcustomkernel ROI.
 CO=3;weight=[((i*3+case)%13-6)/16 for i in range(CO*C*K)];go=[((i*5+case)%17-8)/8 for i in range(CO*S)]
 colgrad=[sum(weight[o*C*K+c*K+k]*go[o*S+i]for o in range(CO))for c in range(C)for k in range(K)for i in range(S)]
 return dict(image=image,offset=offset,mask=mask,colgrad=colgrad,weight=weight,grad_output=go)
def oracle(s,inputs):
 C,G,H,W,KH,KW,OH,OW=[s[k]for k in ['C','G','H','W','KH','KW','OH','OW']];S=OH*OW;K=KH*KW;channel_group=C//G
 im,offset,mask,g=[inputs[k]for k in ['image','offset','mask','colgrad']];out=[0.]*len(g);gi=[0.]*len(im);go=[0.]*len(offset);gm=[0.]*len(mask)
 for c in range(C):
  group=c//channel_group
  for ky in range(KH):
   for kx in range(KW):
    k=ky*KW+kx
    for oy in range(OH):
     for ox in range(OW):
      i=oy*OW+ox;offbase=(group*2*K+2*k)*S+i;maskpos=(group*K+k)*S+i;column=(c*K+k)*S+i
      y=oy*s['SH']-s['PH']+ky*s['DH']+offset[offbase];x=ox*s['SW']-s['PW']+kx*s['DW']+offset[offbase+S];m=mask[maskpos];up=g[column]
      if y<=-1 or y>=H or x<=-1 or x>=W:continue
      val=dyval=dxval=0.
      for iy in [math.floor(y),math.floor(y)+1]:
       for ix in [math.floor(x),math.floor(x)+1]:
        if not(0<=iy<H and 0<=ix<W):continue
        wy=max(0,1-abs(y-iy));wx=max(0,1-abs(x-ix));pos=(c*H+iy)*W+ix;z=im[pos];val+=wy*wx*z
        dwy=1 if y<iy else -1;dwx=1 if x<ix else -1
        dyval+=dwy*wx*z;dxval+=wy*dwx*z;gi[pos]+=up*m*wy*wx
      out[column]=m*val;gm[maskpos]+=up*val;go[offbase]+=up*m*dyval;go[offbase+S]+=up*m*dxval
 return dict(columns=out,grad_image=gi,grad_offset=go,grad_mask=gm)
def loss(s,inputs):return math.fsum(a*b for a,b in zip(oracle(s,inputs)['columns'],inputs['colgrad']))
wrap=lambda a:[777.]*16+list(a)+[777.]*16
if __name__=='__main__':
 assert sys.byteorder=='little';cases=[];prefix=[];tail=[];files={};fd=[]
 for s in map(dims,geometry()):
  inp=data(s);out=oracle(s,inp);s['lengths']=[len(inp[k])for k in ['image','offset','mask','colgrad']];s['output_lengths']=[len(out[k])for k in ['columns','grad_image','grad_offset','grad_mask']]
  prefix +=[s[k]for k in FIELDS]
  for name in ['image','offset','mask','colgrad']:
   vals=inp[name];assert all(f32(x)==x for x in vals);fname=f'case{s["id"]}_{name}_f32.bin';(P/fname).write_bytes(array.array('f',vals).tobytes());files[fname]={'bytes':(P/fname).stat().st_size,'sha256':hashlib.sha256((P/fname).read_bytes()).hexdigest()};prefix+=wrap(vals)
  for name in ['grad_image','grad_offset','grad_mask']:prefix+=wrap([0.]*len(out[name]))
  for name in ['columns','grad_image','grad_offset','grad_mask']:tail+=wrap(out[name])
  s['oracle_outputs_Float32_exactly_representable']=all(f32(x)==x for vals in out.values()for x in vals)
  s['maximum_abs_outputs']={k:max(map(abs,v))for k,v in out.items()};cases.append(s)
  # Independent analytic adjoint oracle cross-check by smooth scalar finite differences.
  base=loss(s,inp)
  for name,outname in [('image','grad_image'),('offset','grad_offset'),('mask','grad_mask')]:
   n=len(inp[name]);indices=sorted(set([0,n//7,n//3,n//2,2*n//3,n-1]))
   for j in indices:
    original=inp[name][j];eps=2**-16;inp[name][j]=original+eps;hi=loss(s,inp);inp[name][j]=original-eps;lo=loss(s,inp);inp[name][j]=original;value=(hi-lo)/(2*eps);analytic=out[outname][j];assert abs(value-analytic)<2e-8,(s['id'],name,j,value,analytic);fd.append({'case':s['id'],'field':name,'index':j,'finite_difference':value,'analytic':analytic,'absolute_error':abs(value-analytic)})
  # Allcolumn gradients arise from actual declaredweight/grad_outputCPUcontraction, not arbitraryopaque target output.
  for name in ['weight','grad_output']:
   fname=f'case{s["id"]}_{name}_f32.bin';(P/fname).write_bytes(array.array('f',inp[name]).tobytes());files[fname]={'bytes':(P/fname).stat().st_size,'sha256':hashlib.sha256((P/fname).read_bytes()).hexdigest()}
 (P/'oracle_f64.bin').write_bytes(array.array('d',prefix+tail).tobytes());files['oracle_f64.bin']={'bytes':(P/'oracle_f64.bin').stat().st_size,'sha256':hashlib.sha256((P/'oracle_f64.bin').read_bytes()).hexdigest()}
 h='#pragma once\nstruct Spec{'+','.join([])+'int '+','.join(FIELDS)+';};\n';h+='constexpr int CASES=4,PREFIX='+str(len(prefix))+',COUNT='+str(len(prefix+tail))+';\n';h+='const Spec SPECS[]={'+','.join('{'+','.join(str(s[k])for k in FIELDS)+'}'for s in cases)+'};\n';(P/'study_params.h').write_text(h)
 jl='const PREFIX='+str(len(prefix))+'\nconst COUNT='+str(len(prefix+tail))+'\nconst CASES=[\n'+',\n'.join('('+','.join(f'{k}={s[k]}'for k in FIELDS)+',)'for s in cases)+'\n]\n';(P/'study_params.jl').write_text(jl)
 for name in ['study_params.h','study_params.jl']:files[name]={'bytes':(P/name).stat().st_size,'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}
 plan={'benchmark':'convolutionDeformable','cases':cases,'input_prefix_count':len(prefix),'artifact_count':len(prefix+tail),'custom_source_device_kernels':3,'kernel_launches':12,'threads':512,'numerical_contract':{'dual_atol':2e-6,'dual_rtol':2e-6,'CPU_atol':2e-6,'CPU_rtol':2e-6,'input_prefix':'exact','guards':'exact'},'ROI':'Sumperbatchsourcebackwardtriplet coordgradient→atomicinputgradient→forwardim2col, wholedevicecompletion. Torchmatmuls/bias/offsetprediction/autograd andlibraryintegration unverified, gradientcolumnsCPUfixtureconstructedoutsideROI.','native_integration_adapted':True,'exclude_untouched_native_aggregates':True,'native_reference_repaired':False,'offset_scope':'Nonintegerdyadic±.25/±.375, minfloorknotmargin.25; masks.25/.5/.75; meaningfulperturbedinputs andfullindependentadjointchecks.','finite_difference_probes':fd,'files':files,'python':sys.version};(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'cases':len(cases),'prefix':len(prefix),'count':len(prefix+tail),'FD_probes':len(fd),'all_outputs_f32_representable':all(s['oracle_outputs_Float32_exactly_representable']for s in cases)}))
