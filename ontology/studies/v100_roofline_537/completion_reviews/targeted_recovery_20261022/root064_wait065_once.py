from pathlib import Path
import json,hashlib,subprocess,datetime,time,os
P=Path('/tmp/roofline-completion-20261021');D=P/'tgvnn064_foreign065_recovery_proposal_v2';receipt=P/'root064_follower_receipt.json';assert not receipt.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(status,**kwargs):receipt.write_text(json.dumps(dict(status=status,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor={'pid':319164,'start_ticks':'83719861'},maximum_collector_invocations=1,**kwargs),indent=2)+'\n')
def alive():
 p=Path('/proc/319164/stat')
 try:
  f=p.read_text().rsplit(')',1)[1].split();return f[19]=='83719861'and f[0]not in ['Z','X']
 except FileNotFoundError:return False
save('waiting_exact065_generation_no_signals')
start=time.monotonic()
while alive():
 if time.monotonic()-start>86400:save('wait_deadline_no_launch');raise SystemExit(0)
 time.sleep(5)
p=D/'controller_foreign_receipt.json'
if not p.exists():save('predecessor_missing_receipt_no_launch');raise SystemExit(0)
d=json.loads(p.read_text())
if d['status']!='finite_traversal_returned_root_admission_required':save('predecessor_not_clean_no_launch',predecessor_status=d['status']);raise SystemExit(0)
assert sha(D/'controller.py')=='4837cb70cd41b92947b672322c3729e285781706db4c0f8454b80c5bcb1eac07';assert sha(D/'launch_review.json')=='087e9285b57945b8f302035edc164775050508f83c62c906d127148c6193f99c'
# Exact same independently reviewed controller, combined all-four selector and
# canonical manifested disposable NEWscratch cleanup. No history restart.
save('launching_one_reviewed_allfour_collector_traversal',predecessor_receipt_sha256=sha(p))
with(P/'root064_controller_stdout.log').open('xb')as log:r=subprocess.run(['/usr/bin/python3',str(D/'controller.py'),'--execute','--group','tgvnn','--review-sha256',sha(D/'launch_review.json')],cwd='/home/imo/HeCBench_julia',stdout=log,stderr=subprocess.STDOUT)
save('finite_call_returned_root_review_required',returncode=r.returncode)
