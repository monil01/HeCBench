# Own CPU report authoring. Does not import proposed code or touch /dev/shm.
from pathlib import Path
import datetime,hashlib,json,os
D=Path('/tmp/roofline-completion-20261021/tmpfs_scratch_optin_proposal')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value): (D/name).write_text(json.dumps(value,indent=2)+'\n')
cgroup_raw=Path('/proc/self/cgroup').read_text(); cgroup=cgroup_raw.split('0::',1)[1].splitlines()[0]
p=Path('/sys/fs/cgroup')/cgroup.lstrip('/'); ancestors=[]
while p!=Path('/sys/fs/cgroup'):
    values={}
    for name in ('memory.max','memory.high','memory.current','memory.swap.max','memory.events'):
        source=p/name;values[name]=source.read_text() if source.is_file() else None
    ancestors.append({'path':str(p),'inode':p.stat().st_ino,'values':values});p=p.parent
capacity={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'uid':os.getuid(),'meminfo':Path('/proc/meminfo').read_text(),
 'self_cgroup':cgroup_raw,'ancestor_memory_records':ancestors,
 'root_memory_max_present':Path('/sys/fs/cgroup/memory.max').exists(),
 'relevant_mountinfo_lines':[x for x in Path('/proc/self/mountinfo').read_text().splitlines() if ' /dev/shm ' in x or ' /sys/fs/cgroup ' in x],
 'filesystems':{},'scope':'Read-only sequential observations, not RAM reservation; future execution must re-read its own cgroup ancestors and memory. No GPU queries or external scratch writes.'}
for name in ('/tmp','/dev/shm'):
    s=os.statvfs(name);capacity['filesystems'][name]={'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize,'free_inodes':s.f_favail}
write('capacity_readonly.json',capacity)
helper=D/'proposed/tools/v100_tmpfs_scratch.py'; collector=D/'proposed/tools/v100_roofline_collect.py'
review={'schema_version':1,'root_reviewed':False,'proposal_only':True,
 'base_collector_sha256':sha(D/'collector_C21_original.py'),
 'adopted_collector_sha256':sha(collector),'helper_sha256':sha(helper),
 'guard_sha256':'a2007386239acd7ba26ebcf64aa211756d759570ec8ee0d1a15ccccc25fbc2d3',
 'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',
 'limits':{'attempt_logical_cap_bytes':12*1024**3,'owned_root_logical_cap_bytes':32*1024**3,
 'host_mem_available_floor_bytes':128*1024**3,'finite_cgroup_headroom_floor_bytes':16*1024**3,
 'tmp_disk_floor_bytes':2*1024**3,'tmpfs_free_floor_bytes':2*1024**3,'maximum_file_count':10000},
 'models':{},'required_model_record_fields':{'original_plan_path':'Existing unchanged exact source-reviewed SU3 plan',
 'original_plan_sha256':'SHA of that existing plan','fresh_plan_sha256':'SHA of future exact scratch-prefix-only fresh plan',
 'owned_root':'/dev/shm/hecbench-v100-uid10384-ROOT_REVIEWED_UNIQUE_ID',
 'configuration_id':'Exact existing CID','raw_config_sha256':'Exact selected config SHA',
 'reference_result_sha256':'Exact selected result SHA','path_roles_independently_reviewed':True,
 'storage_only_no_algorithm_ROI_or_checker_change':True,
 'read_only_bindings':'Full path->SHA map containing all fresh plan execution bindings, selected result/config, helper, adopted collector, guard, SDK/method/selection/source proofs; no mutable output/build target paths'},
 'not_ready_reason':'No owned /dev/shm tree or executable plan created, no root authorization, no fresh staged-source inverse/copy proof yet. Template intentionally rejected by proposed policy.'}
write('root_review_TEMPLATE_NOT_APPROVED.json',review)
write('source_binding_and_review.json',{'base':{'path':'/home/imo/HeCBench_julia/tools/v100_roofline_collect.py','sha256':sha(D/'collector_C21_original.py')},
 'live_base_unchanged_at_delivery':sha('/home/imo/HeCBench_julia/tools/v100_roofline_collect.py')==sha(D/'collector_C21_original.py'),
 'candidate':{'path':str(collector),'sha256':sha(collector)},'helper':{'path':str(helper),'sha256':sha(helper)},
 'diff':{'path':str(D/'minimal.patch'),'sha256':sha(D/'minimal.patch')},
 'source_review_only':True,'imports_of_proposed_code':0,'GPU_queries_or_runs':0,'builds':0,'tests':0,'dev_shm_writes':0,
 'integration_points':['run: prechild and postquiescence source checks; sampled storage checks in same own-session stop loop',
 'collect: opt-in exact review, no-symlink existing700 tmpfs leaf; logs/counters/plan stay /tmp',
 'guard: only exact approved scratch leaf added, same unchanged guard implementation',
 'all child phases: version/build/pre/profile/post receive optional storage monitor',
 'finally: immutable full manifest retained; --retain-auxiliary mandatory, no scratch deletion'],
 'explicit_limits':['User-space sampled watchdog is not a kernel quota or reservation; per-file RLIMIT2GiB remains enforced by existing large-artifact policy.',
 'No exclusive machine reservation; concurrent allocations may force own-session stop.',
 'Tmpfs is volatile. Root must retain/archive full data before shutdown; SHA-only manifests do not recover deleted evidence.',
 'No source-path symlink or broad /dev/shm Landlock access; no RLIMIT_AS imposed on CUDA/Julia.',
 'Initial SU3-only scope. No assets copied, no fresh plan created. Root must source-review exact prefix remap and copy bytes before approval.',
 'No new measurement-method/scientific acceptance; parser, arithmetic, tensor, UUID/PCI, replay and full-five checks remain unchanged.',
 'Default no-opt-in execution branches unchanged; tool source/hash naturally changes on future adoption and requires new controller frontier.']})
write('DELIVERY.json',{'status':'private_source_proposal_requires_root_review_and_fresh_frontier',
 'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{
 str(p.relative_to(D)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(D.rglob('*')) if p.is_file() and p.name not in ('DELIVERY.json','usage_receipt.json')},
 'not_executed':True,'external_scratch_not_created':True,'live_tools_not_changed':True})
print(json.dumps({'delivery_sha256':sha(D/'DELIVERY.json'),'collector_sha256':sha(collector),'helper_sha256':sha(helper),'capacity_mem_available_line':next(x for x in capacity['meminfo'].splitlines() if x.startswith('MemAvailable:'))}))
