#!/usr/bin/env python3
"""Full-file rehash and standalone CPU replay, no GPU invocation."""
import pathlib,argparse,json,hashlib,tempfile,subprocess,math,struct
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;r=json.loads(a.result.read_text());config=json.loads((a.result.parent/'config.json').read_text());assets=config['input_contract']['actual_input_readback_assets'];canonical=B.parent.parent/assets[0]['path'];canonical=canonical.parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
for x in assets:assert sha(B.parent.parent/x['path'])==x['sha256']
artifacts=[];sidecars=[]
for model,m in r['models'].items():
 if not m.get('complete'):continue
 for t in m['trials']:
  p=pathlib.Path(t['output_artifact']['saved_path']);assert p.stat().st_size==15061421*8;artifacts.append(dict(model=model,trial=t['trial'],path=str(p),sha256=sha(p),values=15061421))
  aux=t['dot_history_artifact'];q=pathlib.Path(aux['path']);assert q.stat().st_size==aux['count']*8 and aux['count']%257==0
  actual_dims={2:0,16:0,128:0}
  with q.open('rb')as f:
   while row:=f.read(257*8):
    v=struct.unpack('<257d',row);assert v[0]in actual_dims and all(math.isfinite(x)for x in v);actual_dims[int(v[0])]+=1
  sidecars.append(dict(model=model,trial=t['trial'],path=str(q),sha256=sha(q),bytes=q.stat().st_size,actual_reduction_observation_counts=actual_dims,all_finite=True))
with tempfile.TemporaryDirectory(prefix='minife-CPU-audit-')as d:
 exe=pathlib.Path(d)/'audit';out=pathlib.Path(d)/'replay.json'
 subprocess.run(['g++','-std=c++17','-O2',str(B/'audit_program.cpp'),'-o',str(exe)],check=True)
 subprocess.run([str(exe),str(canonical),str(out)]+[x['path']for x in artifacts],check=True)
 replay=json.loads(out.read_text());assert replay['all_pass']
result=dict(benchmark='miniFE',full_rehash_count=len(artifacts),all_pass=True,canonical_input_assets=assets,all_canonical_input_hashes_match=True,artifacts=artifacts,actual_reduction_sidecars=sidecars,CPU_replay=replay,scope='Full actual-input byte reference, full final arrays, finite postreduction histories and recomputed independent Q1 true residual. Same oracle mathematics reused; no new GPU trial series.')
t=a.output.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2)+'\n');t.replace(a.output);print(json.dumps({k:v for k,v in result.items()if k not in ['canonical_input_assets','artifacts','actual_reduction_sidecars','CPU_replay']},indent=2))
