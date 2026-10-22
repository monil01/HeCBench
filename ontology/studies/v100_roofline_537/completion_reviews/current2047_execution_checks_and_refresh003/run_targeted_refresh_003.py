from pathlib import Path
import subprocess,json,datetime,hashlib,gzip,fcntl
R=Path('/home/imo/HeCBench_julia');P=Path('/tmp/roofline-completion-20261021');S=R/'ontology/studies/v100_roofline_537';D=S/'completion_reviews/targeted_recovery_20261022';D.mkdir(exist_ok=True)
with Path('/tmp/hecbench-roofline-metadata.lock').open()as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);p=S/'integrity_audit.json';dest=D/'before_targeted_refresh003_integrity.json.gz';assert not dest.exists();dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
steps=[('summarize',['python3',str(R/'tools/v100_roofline_summarize.py'),'--source-root',str(R),'--raw-root','/tmp/hecbench-roofline-v100-20261020']),('integrity',['python3',str(R/'tools/v100_roofline_audit.py'),'--source-root',str(R),'--output',str(S/'integrity_audit.json')])]
rows=[]
for name,argv in steps:
 log=P/('targeted_refresh003_'+name+'.log')
 with log.open('xb')as out:p=subprocess.run(argv,cwd=R,stdout=out,stderr=subprocess.STDOUT)
 rows.append({'step':name,'returncode':p.returncode,'log':str(log),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps(rows[-1]),flush=True)
 assert p.returncode==0
p=S/'integrity_audit.json';a=json.loads(p.read_text());assert not a['issues'];report={'status':'targeted_refresh_saved_counter_audit_passed','steps':rows,'audit_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'attempts':a['attempts'],'recomputed_kernels':a['recomputed_kernels'],'issues':0,'live_queues_still_active':True,'scope':'Finishedrawobservations only; no new primary numericalacceptance or wholeROI/causal promotions; Normalize completed raw attempt references into frozen profile identifiers after late collector reports; preserve original exporter hold and all failed histories.'};(P/'targeted_refresh003_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='steps'}),flush=True)
