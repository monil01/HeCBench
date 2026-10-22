#!/usr/bin/env python3
"""Root-adopted CPU staging only. Never compiles, launches, imports target code, or queries a GPU."""
from pathlib import Path
import argparse,json,hashlib,os,datetime
HERE=Path(__file__).resolve().parent
TARGET=Path('/tmp/hecbench-roofline-v100-20261020/batch066-rsc-secp-source-bound')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');ap.add_argument('--root-review',type=Path,required=True);ap.add_argument('--root-review-sha256',required=True);args=ap.parse_args()
 if not args.apply:raise SystemExit('Explicit --apply needed for fresh CPU staging; no GPU execution is implemented')
 if sha(args.root_review)!=args.root_review_sha256:raise RuntimeError('Root review bytes changed')
 review=json.loads(args.root_review.read_bytes());delivery=HERE/'DELIVERY.json'
 if review.get('approved')is not True or review.get('delivery_sha256')!=sha(delivery):raise RuntimeError('Missing exact root packet approval')
 manifest=json.loads(delivery.read_bytes())
 for name,entry in manifest['files'].items():
  p=HERE/name
  if sha(p)!=entry['sha256']or p.stat().st_size!=entry['bytes']:raise RuntimeError('Private delivery changed: '+name)
 proof=json.loads((HERE/'plan_source_proof.json').read_bytes());plans=[]
 if str(TARGET)!=proof['root_reserved_target']:raise RuntimeError('Wrong target root')
 if sha(proof['collector_path'])!=proof['collector_sha256']:raise RuntimeError('Reviewed C21 collector changed')
 if TARGET.exists()or TARGET.is_symlink():raise RuntimeError('Fresh reserved root already exists; preserve it and hold')
 parent=TARGET.parent
 if not parent.is_dir()or parent.resolve()!=parent or parent.stat().st_uid!=os.getuid():raise RuntimeError('Raw-root ownership/path check failed')
 for record in proof['records']:
  p=Path(record['prospective_plan_path']);plan=json.loads(p.read_bytes())
  if sha(p)!=record['prospective_plan_sha256']:raise RuntimeError('Plan changed')
  if plan['model']!='cuda'or plan['benchmark']not in ('rsc','secp256k1'):raise RuntimeError('Scope changed')
  if Path(plan['scratch_cwd'])!=TARGET/plan['benchmark']/'cuda'/'scratch':raise RuntimeError('Nonfresh scratch')
  for path,digest in plan['execution_bindings'].items():
   if sha(path)!=digest:raise RuntimeError('Source/input binding changed: '+path)
  if sha(plan['reference_result'])!=plan['reference_result_sha256']:raise RuntimeError('Reference result changed')
  plans.append((record,plan,p.read_bytes()))
 oldmask=os.umask(0o077)
 try:
  TARGET.mkdir(mode=0o700)
  journal={'status':'staging_started','root_review':str(args.root_review.resolve()),'root_review_sha256':args.root_review_sha256,'delivery_sha256':sha(delivery),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_queries':0,'builds':0,'target_processes':0,'plans':[]}
  def save():
   q=TARGET/'cpu_staging_receipt.json';q.write_text(json.dumps(journal,indent=2)+'\n')
   with q.open('rb')as f:os.fsync(f.fileno())
  save()
  try:
   for record,plan,data in plans:
    slot=TARGET/plan['benchmark']/'cuda';slot.mkdir(parents=True,mode=0o700);scratch=slot/'scratch';scratch.mkdir(mode=0o700)
    for name in ['profile_outputs','ncu_tmp']:
     (scratch/name).mkdir(mode=0o700)
    if plan['benchmark']=='rsc':(scratch/'private_capture').mkdir(mode=0o700)
    (slot/'plan.json').write_bytes(data)
    journal['plans'].append({'path':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'scratch':str(scratch)})
    save()
   for record,plan,data in plans:
    for path,digest in plan['execution_bindings'].items():
     if sha(path)!=digest:raise RuntimeError('Binding changed during CPU staging: '+path)
   journal['status']='CPU_staged_only_no_build_or_GPU';journal['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
  except BaseException as e:
   journal['status']='CPU_staging_failed_preserve_partial_tree';journal['error']=repr(e);save();raise
 finally:os.umask(oldmask)
 print(json.dumps(journal))
if __name__=='__main__':main()
