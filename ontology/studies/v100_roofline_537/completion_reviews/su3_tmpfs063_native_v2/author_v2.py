# Own CPU-only source/JSON authoring; never imports proposed modules.
from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent
V1=Path('/tmp/roofline-completion-20261021/su3_tmpfs063_plan_proposal')
P=Path('/tmp/roofline-completion-20261021/tmpfs_scratch_optin_proposal')
OLD='/tmp/hecbench-roofline-v100-20261020/batch046-su3-native/su3/cuda/scratch'
NEW='/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending/cuda/scratch'
OLDPATH=OLD+'/profile_source/study_native.cu';NEWPATH=NEW+'/profile_source/study_native.cu'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
old=Path(OLDPATH).read_bytes();before=(OLD+'/profile_outputs/private_capture/native_warm_').encode();after=(NEW+'/profile_outputs/private_capture/native_warm_').encode()
assert hashlib.sha256(old).hexdigest()=='4bcc6f1d276dddb360012adbcf4720272d38087aa1c3bf00260f3ea3639a3c1a'
assert old.count(before)==1 and old.count(OLD.encode())==1
new=old.replace(before,after,1);assert new.count(after)==1 and new.replace(after,before,1)==old
(D/'native_copy/original_046_study_native.cu').write_bytes(old)
(D/'native_copy/study_native.cu').write_bytes(new)
newsha=hashlib.sha256(new).hexdigest();oldsha=hashlib.sha256(old).hexdigest()
helper=(P/'proposed/tools/v100_tmpfs_scratch.py').read_text()
function=f'''
# Exactly one root-directed source-path exception. No generic source-hash rewriting.
def normalize_exact_native_failure_overlay(normalized, plan, old_plan, row):
    proof=row.get('native_failure_diagnostic_overlay')
    if plan['model']!='cuda':
        if proof is not None: fail('native exception forbidden on Julia models')
        return normalized
    if not isinstance(proof,dict) or proof.get('explicit_root_approval') is not True:
        fail('native exact failure-path source review required')
    old_prefix={OLD!r}; new_prefix={NEW!r}
    old_source={OLDPATH!r}; new_source={NEWPATH!r}
    old_sha={oldsha!r}; new_sha={newsha!r}
    required={{'old_scratch_prefix':old_prefix,'new_scratch_prefix':new_prefix,
        'original_source_path':old_source,'original_source_sha256':old_sha,
        'fresh_source_path':new_source,'fresh_source_sha256':new_sha,
        'matched_occurrences':1,'only_failure_diagnostic_literal_changed':True,
        'kernel_math_timer_checker_bytes_unchanged':True,
        'retained_completion_proof_copy_sha256_is_historical':True}}
    if any(proof.get(k)!=v for k,v in required.items()): fail('native exception identity differs')
    if (plan['benchmark'],plan['scratch_cwd'],old_plan['scratch_cwd'])!=('su3',new_prefix,old_prefix):
        fail('native exception restricted to exact046-to063 pair')
    if old_plan['execution_bindings'].get(old_source)!=old_sha:
        fail('original source not prospectively bound by exact original plan')
    archive=Path(proof['original_source_archive']); fresh=Path(new_source)
    if row['read_only_bindings'].get(str(archive))!=old_sha or sha(archive)!=old_sha:
        fail('retained original source archive absent/changed')
    if row['read_only_bindings'].get(new_source)!=new_sha or sha(fresh)!=new_sha:
        fail('fresh native source absent/changed')
    if Path(old_source).exists() and sha(old_source)!=old_sha:
        fail('present original source changed')
    original_bytes=archive.read_bytes(); fresh_bytes=fresh.read_bytes()
    before=(old_prefix+'/profile_outputs/private_capture/native_warm_').encode()
    after=(new_prefix+'/profile_outputs/private_capture/native_warm_').encode()
    if original_bytes.count(before)!=1 or original_bytes.count(old_prefix.encode())!=1:
        fail('original failure prefix occurrence differs')
    if fresh_bytes.count(after)!=1 or fresh_bytes.count(new_prefix.encode())!=1:
        fail('fresh failure prefix occurrence differs')
    if original_bytes.replace(before,after,1)!=fresh_bytes or fresh_bytes.replace(after,before,1)!=original_bytes:
        fail('native source differs beyond exact failure-path prefix')
    if normalized['execution_bindings'].get(old_source)!=new_sha:
        fail('only approved fresh source binding may normalize')
    normalized['execution_bindings'][old_source]=old_sha
    return normalized

'''
needle='class Policy:\n';assert helper.count(needle)==1;helper=helper.replace(needle,function+needle)
needle="        if replace_paths(plan,new_scratch,old['scratch_cwd'])!=old:\n            fail('fresh plan must be ONLY exact scratch-prefix substitution')"
replacement="""        normalized=replace_paths(plan,new_scratch,old['scratch_cwd'])
        normalized=normalize_exact_native_failure_overlay(normalized,plan,old,row)
        if normalized!=old:
            fail('fresh plan differs beyond exact scratch prefix and approved native binding')"""
assert helper.count(needle)==1;helper=helper.replace(needle,replacement)
(D/'future_tools/v100_tmpfs_scratch.py').write_text(helper)
(D/'future_tools/v100_roofline_collect_tmpfs.py').write_bytes((V1/'future_tools/v100_roofline_collect_tmpfs.py').read_bytes())
(D/'helper_minimal.patch').write_text(''.join(difflib.unified_diff((P/'proposed/tools/v100_tmpfs_scratch.py').read_text().splitlines(True),helper.splitlines(True),fromfile='v1/v100_tmpfs_scratch.py',tofile='v2/v100_tmpfs_scratch.py')))
(D/'native_source_only_prefix.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile=OLDPATH,tofile=NEWPATH)))
for model in ('cuda','cuda_jl','ka','jacc'):
 p=V1/'symbolic_plans/su3'/model/'plan.json';plan=json.loads(p.read_text())
 if model=='cuda':
  assert plan['execution_bindings'][NEWPATH]==oldsha;plan['execution_bindings'][NEWPATH]=newsha
  restored=json.loads(json.dumps(plan));restored['execution_bindings'][NEWPATH]=oldsha
  assert restored==json.loads(p.read_text())
 dest=D/'collector_plans/su3'/model/'plan.json';save(dest,plan)
 if model!='cuda':assert dest.read_bytes()==p.read_bytes()
proof={'explicit_root_approval':False,'old_scratch_prefix':OLD,'new_scratch_prefix':NEW,
 'original_source_path':OLDPATH,'original_source_sha256':oldsha,'original_source_archive':str(D/'native_copy/original_046_study_native.cu'),
 'fresh_source_path':NEWPATH,'fresh_source_sha256':newsha,'fresh_source_archive':str(D/'native_copy/study_native.cu'),
 'matched_occurrences':1,'only_failure_diagnostic_literal_changed':True,'kernel_math_timer_checker_bytes_unchanged':True,
 'retained_completion_proof_copy_sha256_is_historical':True,
 'inverse_bytes_equal':True,'original_bytes':len(old),'fresh_bytes':len(new),
 'scope':'Only host if(!ok) warmup failure diagnostic destination literal. No kernel/correctness/timer/fixture/guard/output payload changes. Original historical source/proof untouched.'}
save(D/'native_overlay_exact_proof.json',proof)
print(json.dumps({'native_old_sha256':oldsha,'native_new_sha256':newsha,'old_bytes':len(old),'new_bytes':len(new),'helper_v2_sha256':sha(D/'future_tools/v100_tmpfs_scratch.py'),'collector_UNCHANGED_sha256':sha(D/'future_tools/v100_roofline_collect_tmpfs.py')}))
