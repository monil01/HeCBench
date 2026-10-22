from pathlib import Path
import hashlib,json,subprocess,datetime
P=Path('/tmp/roofline-completion-20261021');D=P/'tsne069_spsort070_exact_recovery_proposal';receipt=P/'root069_070_execution_receipt.json';assert not receipt.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'controller.py')=='094e5048bb61dc609ed0bc69898709df17420675b78ddfe249c9adf5b12515d3';assert sha(D/'launch_review.json')=='4d8556df9c05a9628f91ff7e1f81ac7c0465e5f4eeacfc6ce00bde307c3f51f8';assert sha(P/'tsne069_spsort070_independent_review/independent_review.json')=='57bfbedbfbfc08c5573d9c3383a61d7372afbcbbbfccb26bd1023aee68d2040e'
state={'status':'root_reviewed_finite_calls_active','steps':[],'maximum_collector_invocations':2,'GPU':'6only','qualification':'Guard-stop gate after each group CLI. SPSORT CLI can traverse4models; each model stillownsGPUidle/lock/guard, interruptedmodelnotretried. No immediatebetweenmodelgrouphalt claimed.','translation_attempts':0};receipt.write_text(json.dumps(state,indent=2)+'\n')
for group in ['tsne','spsort']:
 log=P/('root069070_'+group+'_stdout.log')
 with log.open('xb')as out:r=subprocess.run(['/usr/bin/python3',str(D/'controller.py'),'--execute','--group',group,'--review-sha256',sha(D/'launch_review.json')],cwd='/home/imo/HeCBench_julia',stdout=out,stderr=subprocess.STDOUT)
 p=D/('controller_'+group+'_receipt.json');d=json.loads(p.read_text())if p.exists()else{};state['steps'].append({'group':group,'returncode':r.returncode,'receipt_status':d.get('status'),'log':str(log)});receipt.write_text(json.dumps(state,indent=2)+'\n')
 if r.returncode or d.get('status')!='finite_traversal_returned_root_admission_required':state['status']='ended_group_hold_no_remaining_calls';break
else:state['status']='finite_calls_returned_root_scientific_review_required'
state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt.write_text(json.dumps(state,indent=2)+'\n')
