#!/usr/bin/env python3
"""Byte-preserving own-study storage consolidation; no source/result/config edits.
Independently generated full captures remain at their original paths. Only after
complete rehash AND full byte comparison may identical files share an inode.
Original creator/path/mtime/size/inode are recorded before the atomic hardlink
replacement. No unrelated paths and no unique bytes are removed.
"""
import argparse,pathlib,json,hashlib,os,datetime
p=argparse.ArgumentParser();p.add_argument('--results',nargs='+',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();R=pathlib.Path(__file__).resolve().parents[2];allowed=(R/'.porting-state/porting_logs/permutate-julia-shared').resolve()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def equal(x,y):
 with x.open('rb')as f,y.open('rb')as g:
  while True:
   b=f.read(8*1024**2);c=g.read(8*1024**2)
   if b!=c:return False
   if not b:return True
captures=[]
for result in a.results:
 r=json.loads(result.read_text())
 assert r['benchmark']=='permutate'
 for model,data in r['models'].items():
  if not data.get('complete') or data.get('status')!='native_check_pass':continue
  for trial in data.get('trials',[]):
   if trial['trial']!=1:continue
   for case in trial['cases']:
    path=pathlib.Path(case['actual_internal_packed_capture']).resolve();assert path.is_relative_to(allowed) and path.is_file() and not path.is_symlink()
    st=path.stat();digest=sha(path);after=path.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    captures.append(dict(path=str(path),model=model,case=case['case'],source_result=str(result.resolve()),result_sha256=sha(result),sha256=digest,original_bytes=st.st_size,original_mtime_ns=st.st_mtime_ns,original_inode=st.st_ino,device=st.st_dev))
groups={};changes=[]
for entry in captures:
 path=pathlib.Path(entry['path']);key=(entry['device'],entry['original_bytes'],entry['sha256']);first=groups.setdefault(key,entry)
 if path==pathlib.Path(first['path']):continue
 source=pathlib.Path(first['path']);st=path.stat()
 if st.st_ino==source.stat().st_ino:continue
 assert equal(source,path),'Full byte comparison failed; no changes permitted'
 rec=dict(capture=entry,identical_source=first,full_byte_comparison=True,operation='atomic_same_path_hardlink_replacement' if a.apply else 'review_only',content_loss=False)
 if a.apply:
  temp=path.with_name(path.name+'.identical-link-'+str(os.getpid()));assert not temp.exists();os.link(source,temp);os.replace(temp,path);assert sha(path)==entry['sha256'] and path.stat().st_ino==source.stat().st_ino
 rec['final_inode']=path.stat().st_ino;changes.append(rec)
out=dict(benchmark='permutate',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),apply=a.apply,scope='Own exact duplicated internal captures only. Every original path and all bytes preserved; independent creation metadata retained. Source/selected results/config/trials unchanged.',captures=captures,changes=changes,unique_duplicate_bytes_consolidated=sum(x['capture']['original_bytes']for x in changes)if a.apply else 0)
a.output.parent.mkdir(exist_ok=True,parents=True);q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({k:v for k,v in out.items()if k not in ['captures','changes']}))
