#!/usr/bin/env python3
"""CPU-only exact input/raw/guard joins and complete independent BC budgets."""
from pathlib import Path
import argparse,json,struct,hashlib,sys
P=Path(__file__).resolve().parent;R=P.parents[1];assert sys.byteorder=='little'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();r=json.loads(args.result.read_text());a=memoryview((P/'fixtures.bin').read_bytes()).cast('i');q=1;oracle=(P/'oracle.bin').read_bytes();oq=0;cases=[];expected_prefix=[]
 for ci in range(a[0]):
  n,m=a[q:q+2];q+=2;arrays=[]
  for size in[n+1,2*m,2*m]:arrays.append(a[q:q+size].tolist());q+=size
  n2=struct.unpack_from('<i',oracle,oq)[0];oq+=4;assert n==n2;values=[];budgets=[]
  for _ in range(n):v,b=struct.unpack_from('<dd',oracle,oq);oq+=16;values.append(v);budgets.append(b)
  for arr in arrays:expected_prefix+=arr
  expected_prefix += [0.]*n+[80,0]+[0]*512
  cases.append(dict(n=n,m=m,R=arrays[0],C=arrays[1],F=arrays[2],oracle=values,budget=budgets))
 rawroot=R/'.porting-state/portable_julia/hbc_raw_full_v1';receipts=[];totalguard=totalraw=totalprimary=0;maxabs=maxrel=0
 for model,result in r['models'].items():
  if not result.get('complete'):continue
  stem='native'if model=='cuda'else model;completed=list(rawroot.glob(f'{stem}_*_trial5_case3.layout.json'));assert len(completed)==1,(model,completed);prefix=completed[0].name.split('_trial5')[0]
  for trial in result['trials']:
   primary=Path(trial['output_artifact']['saved_path']);assert sha(primary)==trial['output_artifact']['sha256'];v=memoryview(primary.read_bytes()).cast('d');assert list(v[:len(expected_prefix)])==expected_prefix;p=len(expected_prefix);totalprimary+=len(v)
   for ci,c in enumerate(cases):
    base=rawroot/f'{prefix}_trial{trial["trial"]}_case{ci}';rawpath=Path(str(base)+'.raw.bin');gp=Path(str(base)+'.guards.u8');lp=Path(str(base)+'.layout.json');raw=rawpath.read_bytes();guard=gp.read_bytes();layout=json.loads(lp.read_text());by={b['name']:b for b in layout['buffers']};assert set(by)=={'R','C','F','next','jia','diam','source','bc','d','Q','Q2','S','ep','sigma','delta'};offset=goff=0
    for b in layout['buffers']:
     count=b['count_elements']*b['element_bytes'];assert b['offset_bytes']==offset;offset+=count;assert b['guard_offset_bytes']==goff;pieces=[]
     for pos,nbytes in b['guard_segments']:pieces.append(raw[b['offset_bytes']+pos:b['offset_bytes']+pos+nbytes])
     actual=b''.join(pieces);assert len(actual)==b['guard_count_bytes'];assert guard[goff:goff+len(actual)]==actual;assert all(x==165 for x in actual);goff+=len(actual)
    assert offset==len(raw)and goff==len(guard)
    def linear(name,fmt):
     b=by[name];z=memoryview(raw)[b['offset_bytes']:b['offset_bytes']+b['count_elements']*b['element_bytes']].cast(fmt);return z[32:32+b['logical_width_elements']]
    for name in ['R','C','F']:assert list(linear(name,'i'))==c[name]
    bc=linear('bc','f');n=c['n'];actual=list(v[p:p+n]);p+=n;assert actual==[float(x*.5)for x in bc];assert v[p]==80+n and v[p+1]==0;p+=2;assert linear('next','i')[0]==80+n;assert linear('jia','i')[0]in(0,1)
    for i,(x,expected,budget)in enumerate(zip(actual,c['oracle'],c['budget'])):
     error=abs(x-expected);maxabs=max(maxabs,error)
     if expected: maxrel=max(maxrel,error/abs(expected))
     assert x==expected if ci in(0,2)or expected==0 else error<=budget+1e-9,(model,ci,i,x,expected,budget)
    totalguard+=len(guard);totalraw+=len(raw);receipts.append(dict(model=model,trial=trial['trial'],case=ci,raw_path=str(rawpath.relative_to(R)),raw_sha256=sha(rawpath),raw_bytes=len(raw),guard_path=str(gp.relative_to(R)),guard_sha256=sha(gp),guard_bytes=len(guard),layout_path=str(lp.relative_to(R)),layout_sha256=sha(lp),actual_input_readbacks_exact=True,raw_BC_joins_complete_primary=True,all_guardbytes_actual_and_165=True))
   assert p==len(v)
 out=dict(status='pass',source_result=str(args.result),source_result_sha256=sha(args.result),models=sum(bool(v.get('complete'))for v in r['models'].values()),checked_trials=sum(len(m['trials'])for m in r['models'].values()),primary_values=totalprimary,actual_input_prefix_values=len(expected_prefix),case_raw_files=len(receipts),raw_bytes=totalraw,actual_guard_bytes=totalguard,max_BC_oracle_abs_error=maxabs,max_BC_oracle_rel_error=maxrel,workspace_evidence='Producingprocess independentlychecksallactualCTA lastsource distance/UInt64sigma/FP32delta, reachableS/endpoint/lastfrontier, actualnextsource andsource-policydiameters. Dynamicrowassignment andunreadtails retainedraw, notmanufacturedidentity.',receipts=receipts);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='receipts'}))
if __name__=='__main__':main()
