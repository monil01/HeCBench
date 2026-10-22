from pathlib import Path
import hashlib,json,subprocess,datetime
P=Path('/tmp/roofline-completion-20261021');D=P/'dispatch073_slu074_finite_proposals';receipt=P/'root073_074_execution_receipt.json';assert not receipt.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'controller.py')=='7d2e5117f24c2dfc9b6d64ecc0be68b1d22a6492a2cc34a33d96a01b745a26c1';assert sha(D/'launch_review.json')=='f734d6c8e74467b8242d98f5bae3a1662bc153c7ba106d84312a2b9cc45223b4';assert sha(P/'dispatch073_slu074_independent_review/independent_review.json')=='f3156f4b138105fd7b977815b49df4ba0e0e97e0f4e4c4a944e82096b9383ce9'
state={'status':'root_reviewed_finite_calls_active','steps':[],'maximum_collector_invocations':2,'GPU':'6only','qualification':'Guard-stop gate after each group CLI. Dispatch CLI can traverse4models; each model stillownsGPUidle/lock/guard, interruptedmodelnotretried. No immediatebetweenmodelgrouphalt claimed.','translation_attempts':0};receipt.write_text(json.dumps(state,indent=2)+'\n')
for group in ['slu','dispatch']:
 log=P/('root073074_'+group+'_stdout.log')
 with log.open('xb')as out:r=subprocess.run(['/usr/bin/python3',str(D/'controller.py'),'--execute','--group',group,'--review-sha256',sha(D/'launch_review.json')],cwd='/home/imo/HeCBench_julia',stdout=out,stderr=subprocess.STDOUT)
 p=D/('controller_'+group+'_receipt.json');d=json.loads(p.read_text())if p.exists()else{};state['steps'].append({'group':group,'returncode':r.returncode,'receipt_status':d.get('status'),'log':str(log)});receipt.write_text(json.dumps(state,indent=2)+'\n')
 if r.returncode or d.get('status')!='finite_traversal_returned_root_admission_required':state['status']='ended_group_hold_no_remaining_calls';break
else:state['status']='finite_calls_returned_root_scientific_review_required'
state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt.write_text(json.dumps(state,indent=2)+'\n')
