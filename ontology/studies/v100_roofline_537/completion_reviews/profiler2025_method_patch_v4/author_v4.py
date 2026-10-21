"""Private text-only source proposal authoring. Never import/run proposed code."""
from pathlib import Path
import json,hashlib,difflib
R=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;V3=O.parent/'v3';P=O/'proposed/tools';P.mkdir(parents=True,exist_ok=True)
h=lambda b:hashlib.sha256(b).hexdigest();before={}
inv=json.loads((V3/'delivery_inventory.json').read_text())
for name,digest in inv.items():
 b=(V3/name).read_bytes()
 if h(b)!=digest:raise ValueError('Frozenv3drift '+name)
 before[str(V3/name)]=h(b)
names=['v100_roofline_collect.py','v100_roofline_summarize.py','v100_roofline_audit.py','v100_profiler_method.py'];modified={n:(V3/'proposed/tools'/n).read_text()for n in names}
for e in json.loads((V3/'authoring_bindings.json').read_text()):
 b=(R/e['file']).read_bytes()
 if h(b)!=e['base_sha256']:raise ValueError('Live source base changed')
 before[str(R/e['file'])]=h(b)
def sub(name,old,new):
 if modified[name].count(old)!=1:raise ValueError('Ambiguous edit '+name+' '+old[:70])
 modified[name]=modified[name].replace(old,new,1)
c='v100_roofline_collect.py'
old="""            bindings(plan)
            if method:method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
            if isolated_review:"""
new="""            if method:
                # run() returns only after its own session is quiescent. Preserve
                # actual raw-byte identity BEFORE any source/SDK post-gate can fail.
                snapshot={'schema_version':1,'captured_at':utc(),
                    'scope':'Actual retained raw counters after source-defined own-session quiescence; no derivation/admission implied.',
                    'profile_process':provenance['profile'],'counter_present':csv_path.is_file()}
                if csv_path.is_file():
                    size_before=csv_path.stat().st_size
                    digest=sha(csv_path)
                    if csv_path.stat().st_size!=size_before:raise RuntimeError('Counter size changed after quiescence')
                    provenance['counters_sha256']=digest
                    snapshot['counter']={'path':str(csv_path),'sha256':digest,'bytes':size_before}
                dump(slot/'counter_quiescence_snapshot.json',snapshot)
                provenance['counter_quiescence_snapshot_sha256']=sha(slot/'counter_quiescence_snapshot.json')
                postcheck={'schema_version':1,'status':'pending','checked_at':utc(),
                    'scope':'Post-profile source and profiler method identity check, separate from process outcome and numerical correctness.',
                    'counter_quiescence_snapshot_sha256':provenance['counter_quiescence_snapshot_sha256'],
                    'positions_publishable':False}
                try:
                    bindings(plan)
                    method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
                    postcheck.update(status='passed',checked_at=utc(),positions_publishable=False)
                except Exception as post_error:
                    postcheck.update(status='rejected',checked_at=utc(),error=repr(post_error),positions_publishable=False)
                    provenance['status']='profile_method_post_profile_validation_hold'
                    raise
                finally:
                    dump(slot/'profiler_method_post_profile_check.json',postcheck)
                    provenance['profiler_method_post_profile_check_sha256']=sha(slot/'profiler_method_post_profile_check.json')
            else:
                bindings(plan)
            if isolated_review:"""
sub(c,old,new)
sub(c,"        if provenance['status'] != 'not_started_replay_method_blocked':", "        if provenance['status'] not in ('not_started_replay_method_blocked','profile_method_post_profile_validation_hold'):")
s='v100_roofline_summarize.py'
sub(s,"                 'profiler_method_preflight.json','profiler_version.log','profiler_version_payload.bin',", "                 'profiler_method_preflight.json','profiler_version.log','profiler_version_payload.bin',\n                 'counter_quiescence_snapshot.json','profiler_method_post_profile_check.json',")
sub(s,"                       'profiler_version.log','profiler_version_payload.bin','profiler_method_preflight.json','derived.json')", "                       'profiler_version.log','profiler_version_payload.bin','profiler_method_preflight.json','derived.json',\n                       'counters.csv','counter_quiescence_snapshot.json','profiler_method_post_profile_check.json')")
g='v100_profiler_method.py'
sub(g," man=json.loads(files['manifest.json']);p=json.loads(files['provenance.json'])", " man=json.loads(files['manifest.json']);p=json.loads(files['provenance.json'])\n # Input-source archive only: never execute or retroactively import these bytes.\n for name,relative in [('calibration_collector_source.py','tools/v100_roofline_collect.py'),('calibration_write_guard_source.py','tools/v100_profile_write_guard.py')]:\n  checked(files[name],man['bindings'][str(Path(man['source_root'])/relative)])\n need(sha(files['calibration_collector_source.py'])==man['collector_sha256'],'calibration collector archive mismatch')")
sub(g," if provenance.get('profile'):\n  argv=provenance['profile']['executed_command'];", """ if provenance.get('profile'):
  snapshot_raw=files['counter_quiescence_snapshot.json']
  need(sha(snapshot_raw)==provenance['counter_quiescence_snapshot_sha256'],'quiescent counter snapshot SHA mismatch')
  snapshot=json.loads(snapshot_raw)
  need(snapshot['profile_process']==provenance['profile'],'snapshot process outcome changed')
  need(snapshot['counter_present']is('counters.csv'in files),'snapshot/raw counter presence mismatch')
  if snapshot['counter_present']:
   raw=files['counters.csv'];item=snapshot['counter']
   need(sha(raw)==item['sha256']==provenance['counters_sha256']and len(raw)==item['bytes'],'quiescent full raw counter bytes changed')
  post_raw=files['profiler_method_post_profile_check.json']
  need(sha(post_raw)==provenance['profiler_method_post_profile_check_sha256'],'post-profile method check SHA mismatch')
  post=json.loads(post_raw)
  need(post['counter_quiescence_snapshot_sha256']==sha(snapshot_raw),'postgate/snapshot join mismatch')
  need(post['status']=='passed','Saved post-profile source/method validation rejected or incomplete; retain attempt without positions')
  argv=provenance['profile']['executed_command'];""")
# Existing audit already validates present counter bytes before the method gate.
# F1 now guarantees a hash for the new post-gate failure path; no tolerance/skip.
# New helper rejects saved rejected post-check; existing audit retains it as a
# per-attempt hold, requiring it not be selected. Audit/allreduce body unchanged.
for name,source in modified.items():(P/name).write_text(source)
# Exact existing sources are copied only as the83-bound calibration input archive.
archive=O/'calibration_input_sources';archive.mkdir(exist_ok=True)
C=Path('/tmp/roofline-completion-20261021/profiler2025_calibration002');manifest=json.loads((C/'source_binding_manifest.json').read_text());archive_manifest=[]
for filename,relative in [('calibration_collector_source.py','tools/v100_roofline_collect.py'),('calibration_write_guard_source.py','tools/v100_profile_write_guard.py')]:
 source=R/relative;b=source.read_bytes();expected=manifest['bindings'][str(source)]
 if h(b)!=expected:raise ValueError('Calibration input source drift '+str(source))
 (archive/filename).write_bytes(b);archive_manifest.append({'name':filename,'original_path':str(source),'sha256':expected,'bytes':len(b),'role':'exact historical manifest-bound input-source archive; no new execution/import'})
(O/'calibration_input_archive.json').write_text(json.dumps({'manifest_sha256':h((C/'source_binding_manifest.json').read_bytes()),'sources':archive_manifest},indent=2)+'\n')
pack=(V3/'pack_method.py').read_text()
old=" for case,spec in m['cases'].items():"
new=""" # Adjacent source archive was independently copied/hash-checked against the
 # unchanged83binding manifest. It is not executed/imported by this packer.
 for name,relative in [('calibration_collector_source.py','tools/v100_roofline_collect.py'),('calibration_write_guard_source.py','tools/v100_profile_write_guard.py')]:
  source=(Path(__file__).parent/'calibration_input_sources'/name).read_bytes()
  gate.checked(source,m['bindings'][str(Path(m['source_root'])/relative)]);f[name]=source
 for case,spec in m['cases'].items():"""
if pack.count(old)!=1:raise ValueError('Packer edit ambiguous')
(O/'pack_method.py').write_text(pack.replace(old,new,1))
# Preserve current live base and full base->v4 patch, plus precise v3->v4 delta.
basepatch=[];delta=[];bindings=[]
for name in names:
 src=R/'tools'/name
 if name=='v100_profiler_method.py':base='';basepath='/dev/null'
 else:base=src.read_text();basepath='a/tools/'+name
 basepatch.extend(difflib.unified_diff(base.splitlines(True),modified[name].splitlines(True),fromfile=basepath,tofile='b/tools/'+name))
 delta.extend(difflib.unified_diff((V3/'proposed/tools'/name).read_text().splitlines(True),modified[name].splitlines(True),fromfile='v3/tools/'+name,tofile='v4/tools/'+name))
 bindings.append({'file':'tools/'+name,'base_sha256':h(base.encode())if base else None,'v3_sha256':h((V3/'proposed/tools'/name).read_bytes()),'v4_sha256':h((P/name).read_bytes())})
delta.extend(difflib.unified_diff((V3/'pack_method.py').read_text().splitlines(True),(O/'pack_method.py').read_text().splitlines(True),fromfile='v3/pack_method.py',tofile='v4/pack_method.py'))
(O/'minimal.patch').write_text(''.join(basepatch));(O/'v3_to_v4.patch').write_text(''.join(delta));(O/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
for p,digest in before.items():
 if h(Path(p).read_bytes())!=digest:raise ValueError('Before/after source drift '+p)
(O/'source_stability.json').write_text(json.dumps({'before_after_input_sha256':before,'all_unchanged':True,'controller002':h((C/'calibrate_controller.py').read_bytes()),'manifest002':h((C/'source_binding_manifest.json').read_bytes()),'proposed_sources_imported_or_executed':False,'GPU_build_tests_shared_mutations':0},indent=2)+'\n')
print(json.dumps({'private_output':str(O),'files':len(bindings),'calibration_inputs':len(archive_manifest),'not_executed':True}))
