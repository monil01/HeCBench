#!/usr/bin/env python3
"""Post-verification byte-preserving hardlinks for OWN immutable private captures.
No primary artifact, runtime fixture, failed capture, other benchmark or path is
removed. Every replacement requires full hash AND full-byte equality. Creation
metadata and all original logical paths survive in the scientific provenance.
"""
from pathlib import Path
import argparse,json,hashlib,os,datetime,uuid
P=Path(__file__).resolve().parent;BASE=Path('/tmp/hecbench-stencil-ports-slu-private')
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def same(a,b):
 with a.open('rb')as x,b.open('rb')as y:
  while True:
   v=x.read(8*1024*1024);w=y.read(8*1024*1024)
   if v!=w:return False
   if not v:return True
a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--output',type=Path,default=P/'private_storage_consolidation.json');z=a.parse_args();d=json.load(open(z.result));assert d['benchmark']=='slu'and d['matched_performance_valid']and all(d['variant_verified'].values());assert not z.output.exists(),'Immutableconsolidationrecordalreadyexists'
paths=[]
for role in ['native_prepare_v2','verified_native','verified_cuda_jl','verified_ka','verified_jacc']:
 folder=BASE/role;assert folder.is_dir()
 for p in sorted(folder.rglob('*')):
  if p.is_file()and p.suffix in ['.f64','.zlib']:
   assert not p.is_symlink()and BASE in p.parents;paths.append(p)
by_size={}
for p in paths:by_size.setdefault(p.stat().st_size,[]).append(p)
groups={}
for size,ps in by_size.items():
 if len(ps)<2:continue
 for p in ps:groups.setdefault((size,sha(p)),[]).append(p)
records=[];logical=0;blocks=0
for (size,h),ps in sorted(groups.items()):
 if len(ps)<2:continue
 anchor=ps[0];assert sha(anchor)==h
 for p in ps[1:]:
  old=p.stat();ac=anchor.stat()
  if(old.st_dev,old.st_ino)==(ac.st_dev,ac.st_ino):continue
  assert old.st_dev==ac.st_dev and sha(p)==h and same(anchor,p)
  before=dict(path=str(p),sha256=h,bytes=size,mtime_ns=old.st_mtime_ns,device=old.st_dev,inode=old.st_ino,links=old.st_nlink,allocated_blocks512=old.st_blocks)
  temp=p.with_name('.'+p.name+'.samebytes-'+uuid.uuid4().hex);os.link(anchor,temp);os.replace(temp,p);after=p.stat();assert after.st_ino==anchor.stat().st_ino and sha(p)==h
  records.append(dict(before=before,after=dict(path=str(p),sha256=h,bytes=size,device=after.st_dev,inode=after.st_ino,anchor=str(anchor)),full_byte_comparison_before_replacement=True,content_unchanged=True,logical_path_preserved=True));logical+=size
  if old.st_nlink==1:blocks+=old.st_blocks
r=dict(benchmark='slu',result=str(z.result.resolve()),result_sha256=sha(z.result),actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),storage_only=True,new_GPU_executions=0,all_logical_paths_and_bytes_preserved=True,scope='Onlycompletedownprivateprepv2/native/CUDAjl/KA/JACC .f64/.zlib. Primarytrialartifacts, runtimefixtures, failedprepv1 andallotherbenchmarks untouched.',original_creation_caveat='Beforemetadataobservednow; actualcapturecreation/sourceprocess/role remainsinoriginalcheckedtrialrecords andboundinstrumentation. Hardlinks do not create a new GPU capture or scientific verdict.',logical_duplicate_bytes_relinked=logical,potential_unique_inode_blocks512_reclaimed=blocks,record_count=len(records),records=records)
z.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'logical_duplicate_bytes_relinked':logical,'records':len(records),'output':str(z.output)}))
