from pathlib import Path
p=Path('/tmp/roofline-completion-20261021/roi_membership_root_merge_v3_preparation/merge_reviews.py')
s=p.read_text().split('\ndef exchange(a,b):')[0]
s=s.replace('archive, whole-directory atomic exchange.', 'archive, explicitly non-atomic two-renames under the metadata lock.')
s=s.replace("'Linux renameat2 RENAME_EXCHANGE whole roi_reviews directory; no sequential fallback'", "'EXPLICIT v4 metadata-lock-serialized journaled two ordinary same-filesystem directory renames; NOT atomic directory exchange'")
s=s.replace("'roi-v3-'", "'roi-v4-'")
s=s.replace("'schema_version':1,'transaction':transaction,'source_root':str(root)", "'schema_version':1,'transaction':transaction,'filesystem_preflight':filesystem_preflight(reviewdir),'source_root':str(root)")
helper='''
def filesystem_preflight(target):
 target=Path(target);parent=target.parent
 need(target.stat().st_dev==parent.stat().st_dev,'Canonical target is a separate mount; same-filesystem rename not proved')
 # Read only the current mount table. Longest enclosing mount is reported, not inferred.
 rows=[]
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  left,right=line.split(' - ',1);fields=left.split();mount=fields[4].replace('\\\\040',' ').replace('\\\\011','\\t').replace('\\\\134','\\\\')
  if target.resolve().is_relative_to(Path(mount)):rows.append((len(mount),{'mountpoint':mount,'filesystem_type':right.split()[0],'mount_options':fields[5],'device_major_minor':fields[2]}))
 need(rows,'Filesystem type could not be read from mountinfo')
 fs=max(rows,key=lambda x:x[0])[1]
 need('ro'not in fs['mount_options'].split(','),'Repository filesystem mounted readonly')
 return dict(fs,device_id=parent.stat().st_dev,target_and_parent_same_filesystem=True,rename_capability='Not executed by preparer; ordinary rename remains a root runtime prerequisite.',readers_require_metadata_lock=True)
'''
s=s.replace('\ndef validate(root,packets,transaction):',helper+'\ndef validate(root,packets,transaction):')
Path('/tmp/roofline-completion-20261021/roi_membership_root_merge_v4_preparation/merge_reviews_v4.py').write_text(s)
