#!/usr/bin/env python3
"""CPU-only rehash and full independent artifact oracle replay; no GPU reruns."""
import argparse,pathlib,json,subprocess,hashlib,tempfile,statistics
P=pathlib.Path(__file__).resolve().parent
q=argparse.ArgumentParser();q.add_argument('result',type=pathlib.Path);q.add_argument('--fixtures',type=pathlib.Path,default=pathlib.Path('/tmp/hecbench-stencil-ports-rbc-fixtures'));q.add_argument('--output',type=pathlib.Path,required=True);a=q.parse_args();d=json.load(open(a.result));records=[]
with tempfile.TemporaryDirectory(prefix='rbc-cpu-replay-')as t:
 binary=pathlib.Path(t)/'audit';subprocess.run(['g++','-O3','-std=c++17',str(P/'audit_capture.cpp'),'-o',str(binary)],check=True)
 for model,m in d['models'].items():
  if not m.get('trials'):continue
  assert m['complete']and len(m['trials'])==5
  times=[r['time_seconds']for r in m['trials']];assert m['mean_seconds']==statistics.mean(times[1:])
  for r in m['trials']:
   assert r['correct']is True
   x=r['output_artifact'];path=pathlib.Path(x['saved_path']);h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==x['sha256'];assert path.stat().st_size==4*x['count']
   p=subprocess.run([str(binary),str(a.fixtures/'actual_inputs.bin'),str(a.fixtures/'independent_oracle.bin'),str(a.fixtures/'oracle_budgets.bin'),str(path)],capture_output=True,text=True,check=True);row=json.loads(p.stdout);assert row['pass'];records.append(dict(model=model,trial=r['trial'],path=str(path),sha256=h,**row))
result=dict(pass_all=True,benchmark='rayleighBenardConvection',result=str(a.result),result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),artifact_count=len(records),all_artifact_hashes_identical=len({x['sha256']for x in records})==1,means_average_trials2through5=True,records=records)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'artifacts':len(records),'output':str(a.output)}))
