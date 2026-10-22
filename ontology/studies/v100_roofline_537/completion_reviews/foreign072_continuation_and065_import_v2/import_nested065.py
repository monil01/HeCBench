#!/usr/bin/env python3
"""Prospective root CPU-only lossless flat raw alias of one finished nested065 failure.
No collector/parser imports, builds, target/GPU queries, ledger writes or old-data edits.
"""
from pathlib import Path
import argparse,json,hashlib,fcntl,os,tempfile,shutil,datetime,gzip
HERE=Path(__file__).resolve().parent
RAW=Path('/tmp/hecbench-roofline-v100-20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,msg):
 if not x:raise RuntimeError(msg)
def load(p):return json.loads(Path(p).read_bytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--authorization-sha256',required=True);a=ap.parse_args()
 need(sha(a.authorization)==a.authorization_sha256,'Root authorization changed');auth=load(a.authorization);need(auth.get('approved')is True and auth.get('delivery_sha256')==sha(HERE/'DELIVERY.json')and auth.get('allow_exact_nested065_lossless_alias')is True,'New exact root alias approval required')
 delivery=load(HERE/'DELIVERY.json')
 for name,e in delivery['files'].items():need(sha(HERE/name)==e['sha256']and(HERE/name).stat().st_size==e['bytes'],'Private packet changed')
 spec=load(HERE/'nested065_import_inventory.json');source=Path(spec['original_slot']);destroot=Path(spec['prospective_flat_alias_root']);dest=Path(spec['prospective_flat_alias_slot']);need(source==RAW/'batch065/atomicSystemWide__cuda_jl/atomicSystemWide/cuda_jl'and destroot==RAW/'batch065-atomicSystemWide__cuda_jl'and dest==destroot/'atomicSystemWide/cuda_jl','Only exact reviewed source/destination allowed')
 need(RAW.resolve()==RAW and RAW.stat().st_uid==os.getuid(),'Raw ownership or symlink changed');need(not source.is_symlink(),'Source symlink forbidden')
 def validate_source():
  need({p.name for p in source.iterdir()}=={e['name']for e in spec['files']},'Original raw inventory changed')
  for e in spec['files']:
   p=source/e['name'];need(p.is_file()and not p.is_symlink()and p.stat().st_size==e['bytes']and sha(p)==e['sha256'],'Raw parent missing/changed: '+e['name'])
  prov=load(source/'profile_provenance.json');plan=load(source/'plan.json');need(prov.get('finished_at')and prov['status']=='profile_process_failed'and prov['profile']['returncode']==-15 and prov['profile']['stop_reason']=='external_gpu_activity_detected_own_profile_stopped','Failed parent execution changed');need(prov['plan_sha256']==sha(source/'plan.json')and prov['counters_sha256']==sha(source/'counters.csv'),'Parent raw hash joins failed');need((plan['benchmark'],plan['model'])==('atomicSystemWide','cuda_jl'),'Wrongcase');need(not(source/'derived.json').exists(),'No derived positions may be invented')
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
  fcntl.flock(lk,fcntl.LOCK_EX if a.apply else fcntl.LOCK_SH);validate_source();need(not destroot.exists()and not destroot.is_symlink(),'Flat alias exists; preserve/hold')
  frozen=Path('/home/imo/HeCBench_julia')/spec['prospective_frozen_profile'];need(not frozen.exists(),'Canonical alias already frozen; no duplicate import')
  for existing in frozen.parent.glob('*/profile_provenance.json.gz'):
   need(hashlib.sha256(gzip.decompress(existing.read_bytes())).hexdigest()!=spec['source_provenance_sha256'],'Same actual parent already frozen under another ID; hold rather than double-count')
  if not a.apply:print(json.dumps({'validated_private_proposal_only':True,'files':spec['file_count'],'bytes':spec['byte_total'],'new_GPU_processes':0,'new_build_processes':0}));return
  stage=Path(tempfile.mkdtemp(prefix='.cpu065_alias_',dir=RAW));journal=stage/'cpu_import_receipt.json';receipt={'status':'preparing_lossless_failed_capture_alias','authorization':str(a.authorization),'authorization_sha256':a.authorization_sha256,'specification_sha256':sha(HERE/'nested065_import_inventory.json'),'tool_sha256':sha(__file__),'source':str(source),'destination_root':str(destroot),'new_GPU_processes':0,'new_build_processes':0,'counter_derivation_performed':False,'raw_status_unchanged':True,'process_unit':'Same one historical065 profiler process, not a new run; nested path was not discovered by original flat scanner.','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};journal.write_text(json.dumps(receipt,indent=2)+'\n')
  try:
   slot=stage/'atomicSystemWide/cuda_jl';slot.mkdir(parents=True,mode=0o700)
   for e in spec['files']:
    q=slot/e['name'];shutil.copyfile(source/e['name'],q);need(sha(q)==e['sha256']and q.stat().st_size==e['bytes'],'Alias copy differs')
    with q.open('rb')as f:os.fsync(f.fileno())
   validate_source();need(not destroot.exists(),'Alias collision; preserve hidden stage');receipt.update(status='lossless_flat_alias_ready_no_new_execution',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=spec['files']);journal.write_text(json.dumps(receipt,indent=2)+'\n')
   with journal.open('rb')as f:os.fsync(f.fileno())
   # Same-filesystem rename publishes the complete alias under metadata EX.
   os.rename(stage,destroot)
   parentfd=os.open(RAW,os.O_RDONLY);os.fsync(parentfd);os.close(parentfd)
  except BaseException as error:
   receipt.update(status='CPU_alias_copy_failed_preserve_hidden_stage',error=repr(error))
   if stage.exists():journal.write_text(json.dumps(receipt,indent=2)+'\n')
   raise
 print(json.dumps({'alias_root':str(destroot),'status':'CPU_raw_alias_only_root_must_run_existing_summarize_and_audit','new_GPU_processes':0,'new_build_processes':0,'original_nested_bytes_preserved':True}))
if __name__=='__main__':main()
