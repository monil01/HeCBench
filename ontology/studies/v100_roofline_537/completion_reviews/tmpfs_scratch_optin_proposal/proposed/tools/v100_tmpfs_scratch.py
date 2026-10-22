"""PROPOSAL ONLY: reviewed SU3 tmpfs storage policy; never creates or removes scratch.
No GPU calls. Import/use only after independent root review and frontier adoption.
The watchdog is sampled, NOT a kernel-enforced byte quota or RAM reservation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

GIB = 1024**3
UUID = 'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
BASE = 'c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
GUARD = 'a2007386239acd7ba26ebcf64aa211756d759570ec8ee0d1a15ccccc25fbc2d3'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()

def fail(message):
    raise RuntimeError('Reviewed tmpfs policy: '+message)

def exact_path(value):
    p = Path(value)
    if not p.is_absolute() or '..' in p.parts: fail('absolute lexical path required')
    for part in [*reversed(p.parents), p]:
        if part.is_symlink(): fail('symlink path component '+str(part))
    if p.resolve() != p: fail('path alias forbidden')
    return p

def mount_record(path):
    candidates=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields=line.split(); sep=fields.index('-'); mount=fields[4]
        if '\\' in mount: continue
        if path==Path(mount) or Path(mount) in path.parents:
            candidates.append((len(mount), {'line':line,'mount':mount,
                'filesystem':fields[sep+1],'device':fields[2]}))
    if not candidates: fail('mount identity unavailable')
    return max(candidates,key=lambda row:row[0])[1]

def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    fail('MemAvailable unavailable')

def cgroups():
    lines=Path('/proc/self/cgroup').read_text().splitlines()
    unified=[line[3:] for line in lines if line.startswith('0::')]
    if len(unified)!=1: fail('unified memory accounting required')
    base=Path('/sys/fs/cgroup'); relative=Path(unified[0].lstrip('/'))
    if '..' in relative.parts: fail('invalid cgroup membership')
    p=base/relative; result=[]
    while p!=base:
        maximum=p/'memory.max'; high=p/'memory.high'; current=p/'memory.current'
        if not all(x.is_file() for x in (maximum,high,current)):
            fail('ancestor memory accounting missing '+str(p))
        used=int(current.read_text()); limits={}
        for key,source in [('max',maximum),('high',high)]:
            value=source.read_text().strip(); limits[key]=None if value=='max' else int(value)
        result.append({'path':str(p),'inode':p.stat().st_ino,'current_bytes':used,
            'limits_bytes':limits,'finite_headrooms_bytes':{
                key:value-used for key,value in limits.items() if value is not None}})
        p=p.parent
    if not result: fail('no accountable non-root cgroup ancestor')
    return {'membership':unified[0],'ancestors':result,
            'root_memory_controller_files_absent':not(base/'memory.max').exists()}

def tree_bytes(root):
    total=0; files=0
    for directory, dirs, names in os.walk(root,followlinks=False):
        for name in dirs+names:
            p=Path(directory)/name; s=p.lstat()
            if stat.S_ISLNK(s.st_mode): fail('scratch symlink forbidden '+str(p))
            if s.st_uid!=os.getuid(): fail('scratch entry owner changed '+str(p))
            if p.stat().st_dev!=root.stat().st_dev: fail('nested mount forbidden')
            if stat.S_ISREG(s.st_mode): total+=s.st_size; files+=1
            elif not stat.S_ISDIR(s.st_mode): fail('special scratch entry forbidden')
    return total,files

def replace_paths(value, old, new):
    if isinstance(value,str): return value.replace(old,new)
    if isinstance(value,list): return [replace_paths(x,old,new) for x in value]
    if isinstance(value,dict):
        out={}
        for k,v in value.items():
            key=replace_paths(k,old,new)
            if key in out: fail('path normalization key collision')
            out[key]=replace_paths(v,old,new)
        return out
    return value

class Policy:
    def __init__(self, review_path, review_sha, plan_path, plan, slot,
                 collector_sha, helper_sha, guard_sha, retain, artifact_policy, reserve):
        self.review_path=exact_path(str(review_path)); self.review_sha=review_sha
        if sha(self.review_path)!=review_sha: fail('review SHA mismatch')
        self.review_bytes=self.review_path.read_bytes(); review=json.loads(self.review_bytes)
        if review.get('root_reviewed') is not True: fail('root approval absent')
        if review['base_collector_sha256']!=BASE or review['adopted_collector_sha256']!=collector_sha:
            fail('collector identity not approved')
        if review['helper_sha256']!=helper_sha or guard_sha!=GUARD or review['guard_sha256']!=GUARD:
            fail('helper/guard identity changed')
        if review['gpu_uuid']!=UUID: fail('device identity changed')
        key=plan['benchmark']+'/'+plan['model']; row=review['models'][key]
        if plan['benchmark']!='su3' or plan['model'] not in ('cuda','cuda_jl','ka','jacc'):
            fail('initial opt-in restricted to SU3 four models')
        if not retain or not artifact_policy or artifact_policy['benchmark']!='su3':
            fail('retain-auxiliary AND unchanged large-artifact policy required')
        if plan.get('runtime_asset_bytes',0)!=0: fail('runtime asset copies need another review')
        if sha(plan_path)!=row['fresh_plan_sha256']: fail('fresh plan differs')
        original=Path(row['original_plan_path'])
        if sha(original)!=row['original_plan_sha256']: fail('original plan changed')
        old=json.loads(original.read_bytes()); new_scratch=plan['scratch_cwd']
        if replace_paths(plan,new_scratch,old['scratch_cwd'])!=old:
            fail('fresh plan must be ONLY exact scratch-prefix substitution')
        if any(row[k]!=plan[k] for k in ('configuration_id','raw_config_sha256','reference_result_sha256')):
            fail('selected contract changed')
        if row['path_roles_independently_reviewed'] is not True: fail('path role review absent')
        self.read_only_bindings=row['read_only_bindings']
        required={**plan['execution_bindings'],plan['reference_result']:plan['reference_result_sha256'],
                  row['raw_config_path']:plan['raw_config_sha256']}
        if any(self.read_only_bindings.get(path)!=digest for path,digest in required.items()):
            fail('review missing complete source/result bindings')
        for path,digest in self.read_only_bindings.items():
            if sha(path)!=digest: fail('reviewed read-only binding changed '+path)
        self.slot=exact_path(str(slot)); self.root=exact_path(row['owned_root'])
        self.scratch=exact_path(new_scratch)
        if self.slot.is_relative_to(Path('/dev/shm')) or not self.slot.is_relative_to(Path('/tmp')):
            fail('counter/log/plan slot must remain on /tmp')
        if self.root.parent!=Path('/dev/shm') or not re.fullmatch(
                'hecbench-v100-uid'+str(os.getuid())+'-[A-Za-z0-9_-]{8,80}',self.root.name):
            fail('owned-root namespace invalid')
        if self.scratch.parent.parent!=self.root or self.scratch.name!='scratch':
            fail('root/unique-attempt/scratch layout required')
        self.identities={}
        for p in (self.root,self.scratch.parent,self.scratch):
            st=p.stat()
            if not p.is_dir() or st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700:
                fail('precreated owner-only700 directories required')
            self.identities[str(p)]=(st.st_dev,st.st_ino,st.st_uid,stat.S_IMODE(st.st_mode))
        self.mount=mount_record(self.scratch)
        if self.mount['mount']!='/dev/shm' or self.mount['filesystem']!='tmpfs': fail('actual tmpfs required')
        self.cgroup_identity=None
        self.policy=review['limits']; self.reserve=reserve; self.started=time.monotonic()
        required=artifact_policy['payload_bytes']+artifact_policy['auxiliary_allowance_bytes']
        if not(required<=self.policy['attempt_logical_cap_bytes']<=12*GIB): fail('SU3 cap outside reviewed bound')
        if not(self.policy['attempt_logical_cap_bytes']<=self.policy['owned_root_logical_cap_bytes']<=32*GIB): fail('aggregate cap outside bound')
        if self.policy['host_mem_available_floor_bytes']<128*GIB: fail('host floor too small')
        if self.policy['finite_cgroup_headroom_floor_bytes']<16*GIB: fail('cgroup floor too small')
        if self.policy['tmp_disk_floor_bytes']<max(2*GIB,reserve): fail('/tmp floor too small')
        if self.policy['tmpfs_free_floor_bytes']<2*GIB: fail('tmpfs floor too small')
        if not(100<=self.policy['maximum_file_count']<=100000): fail('file count bound missing')
        if row['storage_only_no_algorithm_ROI_or_checker_change'] is not True: fail('scope attestation absent')
        self.samples=[]; self.minimums={}; self.initial=self.sample()

    def sample(self):
        if sha(self.review_path)!=self.review_sha: fail('review mutated')
        for name,identity in self.identities.items():
            p=exact_path(name); st=p.stat()
            if (st.st_dev,st.st_ino,st.st_uid,stat.S_IMODE(st.st_mode))!=identity:
                fail('owned directory replaced or permission changed')
        if mount_record(self.scratch)!=self.mount: fail('mount identity changed')
        group=cgroups(); identity=(group['membership'],[(x['path'],x['inode']) for x in group['ancestors']])
        if self.cgroup_identity is not None and identity!=self.cgroup_identity: fail('cgroup identity changed')
        self.cgroup_identity=identity
        size,count=tree_bytes(self.scratch.parent); total,root_count=tree_bytes(self.root)
        remaining=max(0,self.policy['attempt_logical_cap_bytes']-size)
        available=mem_available(); tmp=os.statvfs(self.slot); shm=os.statvfs(self.scratch)
        tmpfree=tmp.f_bavail*tmp.f_frsize; shmfree=shm.f_bavail*shm.f_frsize
        if size>self.policy['attempt_logical_cap_bytes'] or count>self.policy['maximum_file_count']: fail('attempt byte/file cap exceeded')
        if total>self.policy['owned_root_logical_cap_bytes'] or root_count>4*self.policy['maximum_file_count']: fail('aggregate owned cap exceeded')
        if tmpfree<self.policy['tmp_disk_floor_bytes']: fail('/tmp reserve breached')
        if shmfree<remaining+self.policy['tmpfs_free_floor_bytes']: fail('tmpfs capacity reserve insufficient')
        if available<remaining+self.policy['host_mem_available_floor_bytes']: fail('host available-memory reserve insufficient')
        for ancestor in group['ancestors']:
            for headroom in ancestor['finite_headrooms_bytes'].values():
                if headroom<remaining+self.policy['finite_cgroup_headroom_floor_bytes']: fail('finite cgroup reserve insufficient')
        item={'elapsed_seconds':time.monotonic()-self.started,'attempt_logical_bytes':size,
              'owned_root_logical_bytes':total,'tmp_free_bytes':tmpfree,'tmpfs_free_bytes':shmfree,
              'MemAvailable_bytes':available,'cgroup':group}
        for k in ('tmp_free_bytes','tmpfs_free_bytes','MemAvailable_bytes'):
            self.minimums[k]=min(self.minimums.get(k,item[k]),item[k])
        if not self.samples or item['elapsed_seconds']-self.samples[-1]['elapsed_seconds']>=30:
            self.samples.append(item)
        return item

    def verify_bindings(self):
        for path,digest in self.read_only_bindings.items():
            if sha(path)!=digest: fail('reviewed read-only binding changed '+path)

    def summary(self):
        return {'review_path':str(self.review_path),'review_sha256':self.review_sha,
            'scratch':str(self.scratch),'owned_root':str(self.root),'initial':self.initial,
            'minimum_observed':self.minimums,'samples_30_seconds':self.samples,
            'limits':self.policy,'retention':'All fresh files retained; no tmpfs cleanup by collector.',
            'monitor_semantics':'Sampled user-space watchdog, NOT hard quota, RAM reservation or durable storage. Child per-file RLIMIT remains unchanged. No primary raw data removed.'}
