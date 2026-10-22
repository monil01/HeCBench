from pathlib import Path
import json,subprocess,datetime,hashlib,os
P=Path('/tmp/roofline-completion-20261021');D=P/'rsc_secp066_recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((D/'plan_source_proof.json').read_text());records=[]
for r in proof['records']:
 assert sha(Path(proof['collector_path']))==proof['collector_sha256']
 log=P/('root066_'+r['benchmark']+'_collector.log')
 with log.open('xb')as f:p=subprocess.run(r['root_future_launch_argv'],cwd='/home/imo/HeCBench_julia',stdout=f,stderr=subprocess.STDOUT)
 records.append({'benchmark':r['benchmark'],'returncode':p.returncode,'log':str(log),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 (P/'root066_execution_receipt.json').write_text(json.dumps({'status':'finished'if len(records)==2 else'active','records':records},indent=2)+'\n')
