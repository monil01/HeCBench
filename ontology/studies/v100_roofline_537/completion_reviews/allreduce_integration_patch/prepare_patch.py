"""Private source-text preparation only. Does not import or execute candidates."""
from pathlib import Path
import ast,difflib,json,hashlib
ROOT=Path('/home/imo/HeCBench_julia');D=Path(__file__).resolve().parent

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Preimage occurrence '+old[:100])
    return text.replace(old,new,1)

def save(name,text):
    ast.parse(text)
    source=ROOT/'tools'/name;original=source.read_text()
    (D/(name+'.original')).write_text(original)
    (D/name).write_text(text)
    (D/(name+'.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/tools/'+name,tofile='b/tools/'+name)))

p=ROOT/'tools/v100_roofline_summarize.py';s=p.read_text()
s=replace_once(s,'from pathlib import Path\n','from pathlib import Path\nfrom v100_roofline_allreduce_admission import METHOD as ALLREDUCE_METHOD, HOLD as ALLREDUCE_HOLD, read_raw_slot as allreduce_files, validate as validate_allreduce_method\n')
s=replace_once(s,"                 'pre_profile_checks.log','post_profile_checks.log'):","                 'pre_profile_checks.log','post_profile_checks.log',\n                 'adapter_source.py','allreduce_two_device_review.json',\n                 'allreduce_cpu_review.json','allreduce_admission_source.py',\n                 'allreduce_execution_authorization.json'):")
s=replace_once(s,"        runtime_review=applicability.get((benchmark,model))", """        method_admission=None;method_hold=False
        if prov.get('allreduce_two_device_method'):
            try:
                method_admission=validate_allreduce_method(allreduce_files(slot),root)
            except Exception as error:
                method_hold=True
                effective_status=ALLREDUCE_HOLD
                method_admission={'status':'validation_hold','error':str(error),
                    'method_id':prov.get('allreduce_two_device_method',{}).get('method_id'),
                    'positions_publishable':False,'underlying_process_status':prov['status']}
        runtime_review=applicability.get((benchmark,model))""")
s=replace_once(s,"        attempt={'profile_id':str(target.relative_to(root)),'status':prov['status'],", "        attempt={'profile_id':str(target.relative_to(root)),'status':effective_status if prov.get('allreduce_two_device_method') else prov['status'],\n                 'recorded_status':prov['status'],'method_admission':method_admission,")
s=replace_once(s,"        records=json.loads(observation.read_text())['records'] if observation.exists() else []", "        records=json.loads(observation.read_text())['records'] if observation.exists() and not method_hold else []")
s=replace_once(s,"        if review and observation.is_file() and review['configuration_id']==plan['configuration_id']:","        if review and observation.is_file() and not method_hold and review['configuration_id']==plan['configuration_id']:")
s=replace_once(s,"        state.update(profiling_status=effective_status,", """        state.update(profiling_status=effective_status,
            measurement_method=(ALLREDUCE_METHOD if prov.get('allreduce_two_device_method') else 'ordinary_collector_profile_see_scope'),
            profile_method_admission=method_admission,
            process_unit_accounting=(method_admission.get('process_unit_accounting') if method_admission else None),""")
s=replace_once(s,"            roi_coverage=('not_applicable_no_profiled_device_kernels_source_reviewed' if runtime_applicable else", "            roi_coverage=('allreduce_method_validation_hold_no_publishable_positions' if method_hold else\n                          'single_source_marked_rank1_private_add_not_collective_ROI' if prov.get('allreduce_two_device_method') else\n                          'not_applicable_no_profiled_device_kernels_source_reviewed' if runtime_applicable else")
s=replace_once(s,"        if observation.exists():\n            selected=", "        if observation.exists() and not method_hold:\n            selected=")
s=replace_once(s,"                'classification_scope':'Kernel samples only; whole elapsed ROI and causal bottleneck require separate evidence.',", "                'classification_scope':'Kernel samples only; whole elapsed ROI and causal bottleneck require separate evidence.',\n                'measurement_method':state.get('measurement_method','ordinary_collector_profile_see_scope'),\n                'method_admission_status':(state.get('profile_method_admission')or{}).get('status'),\n                'process_unit_accounting':state.get('process_unit_accounting'),")
# Both the common kernel records and DRAM projection-only rows carry the label.
s=s.replace("                        'profiling_status':status,", "                        'profiling_status':status,\n                        'measurement_method':state.get('measurement_method','ordinary_collector_profile_see_scope'),")
s=replace_once(s,"        if not any(x['profile_id']==attempt['profile_id']for x in attempts):\n            attempts.append(attempt)", """        previous=next((x for x in attempts if x['profile_id']==attempt['profile_id']),None)
        if previous is None:
            attempts.append(attempt)
        elif prov.get('allreduce_two_device_method'):
            # Revalidation is fresh metadata, not a new process/capture attempt.
            previous.update(status=effective_status,recorded_status=prov['status'],method_admission=method_admission)""")
save(p.name,s)

p=ROOT/'tools/v100_roofline_audit.py';s=p.read_text()
s=replace_once(s,'import types\n','import types\nfrom v100_roofline_allreduce_admission import validate as validate_allreduce_method\n')
s=replace_once(s,"                if 'derived.json' in files:\n                    report =", """                if prov.get('allreduce_two_device_method'):
                    try:
                        row['allreduce_method_admission']=validate_allreduce_method(files,root)
                    except Exception as error:
                        state=json.loads((study/'benchmarks'/(plan['benchmark']+'.json')).read_text())['models'][plan['model']]
                        selected_completed=(identity in state.get('profile_ids',[]) and state.get('profiling_status')=='profile_process_completed')
                        require(not selected_completed,'Selected allreduce method is not admitted: '+str(error))
                        row['allreduce_method_admission']={'status':'validation_hold','error':str(error),'positions_publishable':False}
                        qualifications.append({'profile_id':identity,'qualification':'Allreduce method validation hold; no publishable positions or freshly verified primary artifacts: '+str(error)})
                if 'derived.json' in files:
                    report =""")
s=replace_once(s,"                row['audit_status'] = 'pass'","                row['audit_status'] = ('pass_frozen_integrity_method_validation_hold' if row.get('allreduce_method_admission',{}).get('status')=='validation_hold' else 'pass')")
save(p.name,s)

# Receipt of exact source-text preimages; neither generated module is executed.
files=[]
for p in sorted(D.iterdir()):
    if p.is_file():files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(D/'patch_preparation_receipt.json').write_text(json.dumps({'preparation':'Source-text patch generation plus ast.parse only; candidate modules not imported/executed.','files':files},indent=2)+'\n')
print('Private source candidates and unified diffs authored; AST inspected only.')
