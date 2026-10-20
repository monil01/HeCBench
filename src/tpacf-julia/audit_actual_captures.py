#!/usr/bin/env python3
"""CPU-only independent audit of immutable numerical/input/guard evidence."""
from pathlib import Path
import argparse,json,hashlib,gzip,struct
import numpy as np
p=Path(__file__).resolve().parent;R=p.parents[1];ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--result',required=True);ap.add_argument('--receipt',required=True);a=ap.parse_args();rp=Path(a.result);r=json.loads(rp.read_text());m=json.loads((p/'data/fixture_manifest.json').read_text());sha=lambda f:hashlib.file_digest(f.open('rb'),'sha256').hexdigest();state=R/'.porting-state/portable_julia/tpacf_primary';cache={};casbytes=0

def blob(h):
 global casbytes
 if h not in cache:
  b=gzip.decompress((state/'CAS'/(h+'.bin.gz')).read_bytes());assert hashlib.sha256(b).hexdigest()==h;cache[h]=b;casbytes+=len(b)
 return cache[h]

def expected_compiled(g,c):
 j=c['njk'];nr=c['nr'];at=j*64;dd=g[:at].reshape(j,64);dr=g[at:at+nr*j*64].reshape(nr,j,64);rr=g[at+nr*j*64:].reshape(nr,64);D=np.zeros((j+1,64),np.int64);C=D.copy();RR=rr.sum(axis=0);D[0]=dd.sum(axis=0);C[0]=dr.sum(axis=(0,1))
 for k in range(1,j+1):D[k]=D[0]-dd[k-1];C[k]=C[0]-dr[:,k-1,:].sum(axis=0)
 return np.concatenate((D.ravel(),C.ravel(),RR))
# Independent collapsed multiplicity check, distinct from source-ordered suffix oracle.
# All physical bins of the full catalog must also equal weighted type-pair counting.
T=len(m['coordinate_types']);matrix=np.array(m['dot_bin_matrix']);independent=[]
for c in m['cases']:
 n=c['n'];ci=c['case'];ids=[[(i*17+ci*3)%T for i in range(n)]]+[[((i*19+k*13+ci*5)%T)for i in range(n)]for k in range(1,c['nr']+1)];counts=[np.bincount(v,minlength=T)for v in ids];g=np.fromfile(p/c['oracle'],dtype='<i8');at=c['njk']*64;dd=g[:at].reshape(c['njk'],64).sum(axis=0);rr=g[(c['nr']+1)*at:].reshape(c['nr'],64)
 for which,actual in [(0,dd)]+[(k,rr[k-1])for k in range(1,c['nr']+1)]:
  h=np.zeros(64,np.int64);freq=counts[which]
  for x in range(T):
   for y in range(x,T):h[matrix[x,y]]+=int(freq[x]*(freq[x]-1)//2 if x==y else freq[x]*freq[y])
  assert np.array_equal(h[:31],actual[:31]);assert int(h.sum())==n*(n-1)//2
 independent.append({'case':ci,'collapsed_multiplicity_oracles':c['nr']+1,'physical_pair_count':n*(n-1)//2})
issues=[];ledgers=[];arts=[];guardsbytes=0;outputs=0
for t in r['models'][a.model]['trials']:
 trial=t['trial'];candidates=sorted((state/a.model).glob(f'*_trial{trial}.inputs.jsonl'));assert len(candidates)==1,'candidate-specificnonce selector required when multiple diagnosticruns exist';lf=candidates[0];gf=lf.with_name(lf.name.replace('.inputs.jsonl','.guards.bin'));rows=[json.loads(s)for s in lf.read_text().splitlines()];prefix=bytearray();at=0
 for c in m['cases']:
  ci=c['case'];cr=[x for x in rows if x['case']==ci];constant=cr.pop(0);assert constant['kind']=='actual_GPUconstant63';actual=blob(constant['sha256']);assert actual==(p/'data/source_boundaries63.bin').read_bytes();prefix.extend(struct.pack('<9d',c['n'],c['nr'],c['njk'],64,29,5,1,10000,1));prefix.extend(struct.pack('<'+str(c['njk'])+'d',*c['jk_sizes']));prefix.extend(actual);seen=set();assert len(cr)==len(c['source_H2D_frame_sequence'])
  for k,(x,want)in enumerate(zip(cr,c['source_H2D_frame_sequence'])):
   assert x['frame']==k and x['sha256']==want and x['bytes']==131072;b=blob(want);assert b==(p/'data/input_frames'/(want+'.bin')).read_bytes()
   if want not in seen:prefix.extend(b);seen.add(want)
 assert len(prefix)==m['prefix_count_total']*8
 raw=gf.read_bytes();pos=0
 for ci in range(3):
  for bid in range(8):
   header=struct.unpack_from('<3i',raw,pos);pos+=12;assert header==(ci,bid,512);b=raw[pos:pos+512];pos+=512;assert b==bytes([0xa5])*512;guardsbytes+=512
 assert pos==len(raw)
 art=t['output_artifact'];f=Path(art['saved_path']);assert sha(f)==art['sha256'];v=np.fromfile(f,dtype='<f8');assert len(v)==1188674;assert v[:m['prefix_count_total']].tobytes()==prefix;off=m['prefix_count_total']
 for c in m['cases']:
  hdr=v[off:off+4];off+=4;ci=c['case'];g=np.fromfile(p/c['oracle'],dtype='<i8');comp=expected_compiled(g,c);reportN=58+160*(c['njk']+1);assert np.array_equal(hdr,[ci,len(g),len(comp),reportN]);actual=v[off:off+len(g)];assert np.array_equal(actual,g);off+=len(g);assert np.array_equal(v[off:off+len(comp)],comp);off+=len(comp);rep=v[off:off+reportN].reshape(-1,2);off+=reportN;assert np.isfinite(rep).all();assert np.isin(rep[:,0],[-1,0,1,2]).all();assert rep[0,0]==-1 and rep[-4,0]==1;outputs+=len(g)+len(comp)+reportN
 assert off==len(v)
 ledgers.append({'trial':trial,'path':str(lf),'sha256':sha(lf),'rows':len(rows),'guard_path':str(gf),'guard_sha256':sha(gf)});arts.append({'trial':trial,'path':str(f),'sha256':art['sha256']})
receipt={'model':a.model,'result':str(rp),'result_sha256':sha(rp),'actual_input_prefix_values':m['prefix_count_total'],'input_rows_per_trial':len(rows),'complete_actual_input_CAS_frames_rehashed':len(cache),'unique_actual_input_CAS_bytes_rehashed':casbytes,'actual_guard_bytes_all5':guardsbytes,'full_numeric_output_fields_checked_all5':outputs,'independent_full_catalog_pair_oracle_controls':independent,'source_report_precision':'Allnumericvalues/kindmaskretained atactualsource6decimalprecision; observervalidatesallcounts/Wroundingformula. No rawASCIIwhitespace/processstdoutidentityclaim.','input_ledgers_all5':ledgers,'primary_artifacts_all5':arts,'issues':issues,'scope':'Complete source finalhistograms/compiledsourcepolicy/reportnumericfields, notpackedinternal256MBpairassignmentworkspace. Explicitcorrectedhost/paddingreference andknownone-sidedJK/reportpolicylimitations retained.'};Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k not in ['input_ledgers_all5','primary_artifacts_all5']}))
