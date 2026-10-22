from pathlib import Path
import subprocess,json,hashlib,datetime,time
P=Path('/tmp/roofline-completion-20261021');D=P/'tgvnn064_foreign065_recovery_proposal_v2';f=P/'root064_stable_idle_release_receipt.json';assert not f.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(d):f.write_text(json.dumps(d,indent=2)+'\n')
assert json.loads((P/'root064_follower_receipt.json').read_text())['status']=='predecessor_not_clean_no_launch';assert not(D/'controller_tgvnn_receipt.json').exists();assert sha(D/'controller.py')=='4837cb70cd41b92947b672322c3729e285781706db4c0f8454b80c5bcb1eac07';assert sha(D/'launch_review.json')=='087e9285b57945b8f302035edc164775050508f83c62c906d127148c6193f99c'
state={'status':'root_explicit_stable_idle_gate','samples':[],'oldfollower_zero_launch':True,'maximum_collector_invocations':1,'GPU':'6only','approval_basis':'Root independently releases previously unlaunched064 after stable-idle gate; exact independently reviewed controller/source/selectors/deadlines unchanged. No065restart or atomicretry.'};save(state)
for i in range(13):
 r=subprocess.run(['nvidia-smi','-i','6','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True);v=[s.strip()for s in r.stdout.strip().split(',')];clean=r.returncode==0 and len(v)==3 and v[0]=='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'and v[1:] == ['0','0'];state['samples'].append({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query_returncode':r.returncode,'gpu6_row':r.stdout.strip(),'idle':clean});save(state)
 if not clean:state['status']='initial_idle_gate_hold_no_launch_no_retry';save(state);raise SystemExit(0)
 if i<12:time.sleep(5)
state['status']='stable60seconds_launching_one_allfour_traversal';save(state)
with(P/'root064_stable_idle_controller_stdout.log').open('xb')as out:r=subprocess.run(['/usr/bin/python3',str(D/'controller.py'),'--execute','--group','tgvnn','--review-sha256',sha(D/'launch_review.json')],cwd='/home/imo/HeCBench_julia',stdout=out,stderr=subprocess.STDOUT)
state.update(status='finite_call_returned_root_scientific_review_required',returncode=r.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state)
