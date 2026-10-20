#!/usr/bin/env python3
"""Full-byte-proven hardlink consolidation, completed owned captures only.
Every original independently created logical capture path and byte remains available.
Never run while the specified model process is active. No unrelated evidence cleanup.
"""
import argparse,hashlib,json,os,pathlib,datetime
P=pathlib.Path(__file__).resolve().parent
CAPTURE_ROOT=pathlib.Path('/tmp/hecbench-portable-infrastructure-sss-captures')
def digest(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identical(a,b):
 with a.open('rb')as x,b.open('rb')as y:
  while True:
   u=x.read(4*1024*1024);v=y.read(4*1024*1024)
   if u!=v:return False
   if not u:return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=pathlib.Path,required=True);ap.add_argument('--model',choices=['cuda','cuda_jl','ka','jacc'],required=True);ap.add_argument('--completed-capture-dir',type=pathlib.Path,required=True);a=ap.parse_args();r=json.loads(a.result.read_text());m=r['models'][a.model];assert m.get('complete')and len(m.get('trials',[]))==5 and m.get('provenance_valid')is True
 cap=a.completed_capture_dir.resolve();assert cap.parent==CAPTURE_ROOT and cap.is_dir();files=[p for p in cap.rglob('*')if p.is_file()];files +=[pathlib.Path(t['output_artifact']['saved_path'])for t in m['trials']];seen={};rows=[];saved=0
 for p in sorted(set(files),key=str):
  s=p.stat();h=digest(p);row={'path':str(p),'sha256':h,'bytes':s.st_size,'original_independent_file_stat':{'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}};key=(s.st_size,h)
  if key not in seen:seen[key]=p;row['action']='retain_representative'
  else:
   q=seen[key];assert identical(p,q),'SHA collision or concurrent mutation';row.update(representative_path=str(q),full_byte_comparison_passed=True)
   if s.st_ino==q.stat().st_ino and s.st_dev==q.stat().st_dev:row['action']='already_same_inode'
   else:
    tmp=p.with_name(p.name+'.verified-hardlink.tmp');assert not tmp.exists();os.link(q,tmp);os.replace(tmp,p);assert digest(p)==h and identical(p,q);row['action']='byte_identical_hardlink_replacement';saved+=s.st_size
  rows.append(row)
 out=P/'storage_provenance';out.mkdir(exist_ok=True);proof={'benchmark':'sss','run_id':r['run_id'],'model':a.model,'result_path':str(a.result),'result_sha256':digest(a.result),'completed_model_only':True,'independently_created_capture_paths_preserved':True,'logical_content_unchanged':True,'full_byte_and_SHA_checks':True,'no_GPU_execution_or_result_source_acceptance_change':True,'physical_duplicate_bytes_relinked':saved,'rows':rows,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':digest(pathlib.Path(__file__))};dest=out/(r['run_id']+'_'+a.model+'_hardlinks.json');assert not dest.exists();dest.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'proof':str(dest),'physical_duplicate_bytes_relinked':saved,'paths':len(rows)}))
if __name__=='__main__':main()
