"""Independent byte-level coordinate oracle and explicit original-lane model."""
from pathlib import Path
import json,hashlib,math
R=Path(__file__).resolve().parents[2];B=Path(__file__).parent;S=R/'.porting-state/portable_julia/runs';MASK=(1<<64)-1

def fixture(n):
 vals=[]
 for i in range(n):
  x=((i+1)*0x9e3779b97f4a7c15)&MASK;x ^= x>>30;x=(x*0xbf58476d1ce4e5b9)&MASK;x ^= x>>27;vals.append(x)
 return vals

def expected(shapes,original_bug=False):
 inputs=[];outputs=[];descriptors=[]
 for nr,nc in shapes:
  vals=fixture(nr*nc)
  for width in [1,2,4,8]:
   host=[(x&((1<<(8*width))-1)).to_bytes(width,'little') for x in vals]
   for rm in [True,False]:
    for ar in [True,False]:
     gold=[]
     for i in range(nr*nc):
      row,col=divmod(i,nc) if rm else (i%nr,i//nr)
      sr,sc=(nr-row-1,col) if ar else (row,nc-col-1)
      j=sr*nc+sc if rm else sc*nr+sr;gold.append(host[j])
     if original_bug:
      ratio=16//width
      while ratio>1 and ((nc if rm else nr)*width)%(ratio*width):ratio//=2
      out=[None]*(nr*nc);length=((nr+1)//2)*nc if ar else nr*((nc+1)//2)
      for idx in range(0,length,ratio):
       if not rm and not ar:sr=idx%nr;sc=idx//nr;src=idx;dst=(nc-sc-1)*nr+sr
       elif not rm and ar:mod=(nr+1)//2;sr=idx%mod;sc=idx//mod;src=sc*nr+sr;dst=sc*nr+nr-sr-ratio
       elif rm and not ar:mod=(nc+1)//2;sr=idx//mod;sc=idx%mod;src=sr*nc+sc;dst=sr*nc+nc-sc-ratio
       else:sr=idx//nc;sc=idx%nc;src=idx;dst=(nr-sr-1)*nc+sc
       # Original loop reverses every symmetric pair twice, so both vectors retain lane order.
       out[dst:dst+ratio]=host[src:src+ratio];out[src:src+ratio]=host[dst:dst+ratio]
      assert all(x is not None for x in out);gold=out
     prefix=b'\xa5'*32+b''.join(host)+b'\xa5'*32;answer=b'\xa5'*32+b''.join(gold)+b'\xa5'*32
     inputs.append(prefix);outputs.append(answer);descriptors.append({'nr':nr,'nc':nc,'width':width,'row_major':rm,'along_rows':ar,'bytes':len(answer)})
 return b''.join(inputs+outputs),sum(map(len,inputs)),descriptors

def audit():
 rows=[];sums={}
 for kind,rid,shapes in [('scalar','20261018T034310681773Z_reverse2D',[(17,19),(33,35)]),('corrected_vector','20261018T034324442749Z_reverse2D',[(32,64),(64,96)])]:
  p=S/rid/'result.json';d=json.loads(p.read_text());gold,prefix,cases=expected(shapes);assert d['matched_performance_valid']
  for m,x in d['models'].items():
   for t in x['trials']:
    a=t['output_artifact'];data=Path(a['saved_path']).read_bytes();assert data==gold;assert hashlib.sha256(data).hexdigest()==a['sha256'];rows.append({'configuration':kind,'model':m,'trial':t['trial'],'pass':True,'bytes':len(data),'exact_prefix_bytes':prefix,'sha256':a['sha256']})
   assert len(x['trials'])==5 and math.isclose(sum(t['time_seconds'] for t in x['trials'][1:])/4,x['mean_seconds'],rel_tol=1e-11)
  sums[kind]={'configuration_id':d['configuration_id'],'result_path':str(p.relative_to(R)),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':cases,'mean_seconds':{m:x['mean_seconds'] for m,x in d['models'].items()},'oracle_sha256':hashlib.sha256(gold).hexdigest()}
 diag=json.loads((S/'20261018T033733722141Z_reverse2D/result.json').read_text());broken,prefix,_=expected([(32,64),(64,96)],True);ideal,_,_=expected([(32,64),(64,96)]);diff=sum(a!=b for a,b in zip(broken[prefix:],ideal[prefix:]));assert diff==244856
 for t in diag['models']['cuda']['rejected_trials']:assert Path(t['output_artifact']['saved_path']).read_bytes()==broken
 output={'independent_oracle':'Python typed-byte coordinate gather, completeactualinput/output/guardarrays; originalnative diagnostic additionallymatchesexplicitdouble-swaplane model. NotcopiedGPUresults.','accepted_checks':rows,'configurations':sums,'original_vector_diagnostic':{'rejected_captures':5,'original_lane_model_matches_all':True,'wrong_output_bytes':diff,'original_guards_and_input_exact':broken[:prefix]==ideal[:prefix]},'hard_issues':[]}
 (B/'output_audit.json').write_text(json.dumps(output,indent=2)+'\n');print({'accepted_captures':len(rows),'rejected_original_captures':5,'wrong_original_bytes':diff,'hard_issues':0})
if __name__=='__main__':audit()
