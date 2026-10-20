#!/usr/bin/env python3
"""CPU-only complete SHA/prefix/scientific replay. Run after GPU timing processes."""
from pathlib import Path
import argparse,json,hashlib,subprocess,statistics,datetime
P=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def equal_prefix(p,q,n):
 with p.open('rb')as a,q.open('rb')as b:
  while n:
   k=min(n,8*1024*1024);x=a.read(k);y=b.read(k)
   if len(x)!=k or len(y)!=k or x!=y:return False
   n-=k
 return True
def main():
 a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--fixtures',type=Path,default=Path('/tmp/hecbench-stencil-ports-slu-fixtures-v2'));a.add_argument('--replay',type=Path,default=Path('/tmp/hecbench-stencil-ports-slu-replay'));a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=json.load(open(z.result));assert d['benchmark']=='slu';prep=json.load(open(z.fixtures/'native_preparation_report.json'));prefix=z.fixtures/'canonical_actual_input_prefix.bin';assert prefix.stat().st_size==8*prep['input_prefix_count'];rows=[]
 for m in json.load(open(z.result.resolve().with_name('config.json')))['models']:
  v=d['models'][m]
  assert v['complete']and v['provenance_valid']and len(v['trials'])==5
  assert v['mean_seconds']==statistics.mean(t['time_seconds']for t in v['trials'][1:])
  for t in v['trials']:
   assert t['correct'];rec=t['output_artifact'];p=Path(rec['saved_path']);assert rec['dtype']=='float64'and rec['byte_order']=='little';assert p.stat().st_size==8*rec['count'];actual=sha(p);assert actual==rec['sha256'];assert equal_prefix(p,prefix,prefix.stat().st_size)
   process=subprocess.run([str(z.replay.resolve()),str(p),str(z.fixtures/'cases.txt')],capture_output=True,text=True);assert process.returncode==0,(m,t['trial'],process.stderr);oracle=json.loads(process.stdout);assert oracle['pass']is True and oracle['checked_values']==rec['count'];assert all(c['pass']for c in oracle['cases'])
   rows.append(dict(model=m,trial=t['trial'],artifact=str(p),artifact_sha256=actual,complete_input_prefix_byte_equal=True,full_independent_oracle=oracle))
 out=dict(benchmark='slu',pass_all=True,artifact_count=len(rows),result=str(z.result.resolve()),result_sha256=sha(z.result),replay_source_sha256=sha(P/'replay_full_state.cpp'),independent_oracle_source_sha256=sha(P/'independent_oracle.h'),canonical_prefix_sha256=sha(prefix),full_captured_input_values=prep['input_prefix_count'],full_captured_output_values=prep['full_output_count'],rows=rows,native_reference_repaired=True,qualification='Conditionalsharedorderingreference; backwarderrorcontract only, notconditioning/forwardaccuracy/circuitphysics. SourcewholeGPUworkspacecheckedandlosslesslycapturedseparatelybeforeandaftereachtrial; nocanary/safetyclaim.',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());z.output.parent.mkdir(parents=True,exist_ok=True);tmp=z.output.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(z.output);print(json.dumps({'pass_all':True,'artifacts':len(rows),'output':str(z.output)}))
if __name__=='__main__':main()
