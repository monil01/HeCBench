#!/usr/bin/env python3
"""Coordinator commit helper: require dual-reference evidence and unchanged kernel source.
Does not run benchmarks. Shared reference work and each variant are separate commits.
"""
import argparse,json,math,subprocess,sys
from pathlib import Path
from study_evidence import effective_config
import verify_portable_julia as verifier
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--result',type=Path,action='append',required=True);p.add_argument('--variant',choices=['KA','JACC']);p.add_argument('--shared',action='store_true');p.add_argument('--reviewed-dependency-change',default='');p.add_argument('--performance-tuning',action='store_true');p.add_argument('--evidence-refresh',action='store_true');p.add_argument('--measurement-repair',action='store_true');p.add_argument('--variant-archive',type=Path,action='append',default=[]);a=p.parse_args()
if bool(a.variant)==bool(a.shared):p.error('choose exactly one of --variant or --shared')
if a.shared and a.variant_archive:p.error('variant archives belong in their variant commit')
if a.measurement_repair and (a.evidence_refresh or a.performance_tuning):p.error('measurement repair is a separate phase from evidence-only refresh and tuning')
if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip():raise SystemExit('Index already contains staged work; inspect it before using coordinator helper.')
data=[]
for path in a.result:
 if not path.is_absolute():path=root/path
 d=json.loads(path.read_text());c=effective_config(path,json.loads(path.with_name('config.json').read_text()))
 if d['benchmark']!=a.benchmark or not d.get('reference_agreement'):raise SystemExit('Benchmark or reference agreement invalid')
 execution_only=c.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only'
 if len(d.get('reference_checks',[]))!=5 or not all(x.get('status')==('execution_only_pass' if execution_only else 'pass') for x in d['reference_checks']):raise SystemExit('Reference agreement requires five explicit passing comparisons')
 data.append((d,c,path))
parent=root/'src'/(a.benchmark+'-julia');key={'KA':'ka','JACC':'jacc'}.get(a.variant)
def allowed(f):return f.is_file() and (f.suffix in ['.jl','.cu','.cuh','.h','.hpp','.cpp','.toml','.md','.json','.sh','.py'] or f.name in ['Makefile','original_Makefile','legacy_Makefile','LICENSE','COPYING','NOTICE'])
if a.shared:
 for d,c,path in data:
  for model in ['cuda','cuda_jl']:
   if not d['models'][model].get('complete') or d['models'][model].get('status')!='native_check_pass':raise SystemExit('Reference needs all five correct checks; complete artifact capture alone is not verification')
   if verifier.digest_tree(verifier.discover(a.benchmark)[model])!=d['models'][model]['source_hash']:raise SystemExit('Reference source changed since verification: '+model)
 paths=[f for f in parent.iterdir() if allowed(f)]+[f for f in (parent/(a.benchmark+'-julia-cuda.jl')).rglob('*') if allowed(f)]
 for _,config,_ in data:
  for spec in config['models'].values():
   dependency_cwd=Path(spec.get('cwd',str(root)))
   if not dependency_cwd.is_absolute():dependency_cwd=root/dependency_cwd
   for name in spec.get('source_dependencies',[]):
    f=Path(name)
    if not f.is_absolute():f=dependency_cwd/f
    f=f.resolve()
    if f.is_relative_to(parent) and f.is_file() and allowed(f) and not any(part in [a.benchmark+'-julia-KA',a.benchmark+'-julia-JACC'] for part in f.parts):paths.append(f)
 paths=list(dict.fromkeys(paths))
 title='Align '+a.benchmark+' study fixtures and verified references'
else:
 child=parent/(a.benchmark+'-julia-'+a.variant);proof=[]
 scoped_children=[child]
 for _,c,_ in data:
  execution_cwd=Path(c['models'][key].get('cwd',str(child)))
  if not execution_cwd.is_absolute():execution_cwd=root/execution_cwd
  execution_cwd=execution_cwd.resolve()
  if execution_cwd!=child.resolve() and execution_cwd.is_relative_to(parent.resolve()) and execution_cwd.name==child.name:
   scoped_children.append(execution_cwd)
 scoped_children=list(dict.fromkeys(scoped_children))
 for d,c,path in data:
  if not d['variant_verified'].get(key):raise SystemExit('Variant not verified against both references')
  for ref in ('cuda','cuda_jl'):
   comparison=d.get('comparisons',{}).get(key,{}).get(ref,{})
   if comparison.get('status') not in (('execution_only_pass',) if c.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only' else ('direct_artifact_pass','numeric_summary_pass')) or len(comparison.get('trials',[]))!=5 or not all(x.get('status')==('execution_only_pass' if c.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only' else 'pass') for x in comparison['trials']):raise SystemExit('Each reference requires five explicit passing target comparisons; own checks cannot override mismatch')
  if verifier.digest_tree(child)!=d['models'][key]['source_hash']:raise SystemExit('Variant source changed since verification')
  changed=[]
  for model,m in d['models'].items():
   for name,sha in m.get('adapter_file_hashes',{}).items():
    f=Path(name)
    if not f.exists() or verifier.hashlib.sha256(f.read_bytes()).hexdigest()!=sha:changed.append(name)
  if changed and not a.reviewed_dependency_change:raise SystemExit('Dependency drift requires recorded review: '+str(changed))
  for m in d['models'].values():
   if m.get('mean_seconds') is not None:
    expected=sum(t['time_seconds'] for t in m['trials'][1:])/4
    if len(m['trials'])!=5 or not all(t['correct'] for t in m['trials']) or not math.isclose(expected,m['mean_seconds'],rel_tol=1e-11):raise SystemExit('Five-trial evidence invalid')
  proof.append({'numerical_agreement_kind':c.get('numerical_agreement_kind','declared_numeric_artifact_contract'),'numerical_agreement_claim':c.get('numerical_agreement_kind')!='no_numerical_outputs_execution_only','raw_output_identity_claim':False if c.get('numerical_agreement_kind') in ('property_equivalence_not_identical_raw_values','no_numerical_outputs_execution_only') else None,'native_reference_repaired':bool(c.get('native_reference_repaired')),'verification_contract_revised':bool(c.get('verification_contract_revised')),'native_behavior_known_scientific_defect':bool(c.get('native_behavior_known_scientific_defect')),'reference_provenance':c.get('reference_provenance'),'configuration_classification_adjudication':c.get('configuration_classification_adjudication'),'reported_metric_definition_erratum':c.get('reported_metric_definition_erratum'),'untouched_native_aggregate_eligible':c.get('numerical_agreement_kind')!='no_numerical_outputs_execution_only' and str(c.get('gpu',6))=='6' and not c.get('verification_contract_revised') and not c.get('native_reference_repaired') and not c.get('exclude_untouched_native_aggregates',False) and not c.get('exclude_from_scientific_fidelity_aggregates',False) and not c.get('native_behavior_known_scientific_defect',False),'run_id':d['run_id'],'configuration_id':c['configuration_id'],'protocol':d['protocol'],'variant_verified':True,'reference_agreement':True,'evidence_strength':c.get('numerical_evidence_strength') or ('aggregate_count_and_error_sum_only' if a.benchmark=='feynman-kac' else 'full_output_artifacts'),'comparisons':d['comparisons'][key],'models':d['models'],'ontology_context':d.get('ontology'),'input_contract':d.get('input_contract'),'input_prefix_count':d.get('input_prefix_count',0),'timing_region_metadata_adjudication':d.get('timing_region_metadata_adjudication'),'prefix_validation_transition':d.get('prefix_validation_transition'),'declared_execution_model_spec':c['models'][key],'execution_source_bindings':d['models'][key].get('adapter_file_hashes'),'attempt_accounting':d.get('attempt_accounting'),'token_accounting':d.get('token_accounting'),'reviewed_dependency_change':{'reason':a.reviewed_dependency_change,'paths':changed} if changed else None,'raw_result':str(path.relative_to(root))})
 receipt={'benchmark':a.benchmark,'variant':'julia-'+key,'configurations':proof,
          'execution_source_scope':'Canonical source hash and explicit executed source dependency bindings are separate. Secondary method entry points retain their scoped directories and configurations.'}
 for owned_child in scoped_children:owned_child.joinpath('verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 paths=list(dict.fromkeys(f for owned_child in scoped_children for f in owned_child.rglob('*') if allowed(f)));title=('Remeasure ' if a.measurement_repair else 'Link verified ' if a.evidence_refresh else 'Tune verified ' if a.performance_tuning else 'Add verified ')+a.benchmark+' '+a.variant+(' primary evidence' if a.evidence_refresh else ' implementation')
# Retain a rejected candidate with its model commit without moving it into the
# source-hashed execution directory or mixing it into shared reference work.
archives=[]
for name in a.variant_archive:
 f=(name if name.is_absolute() else root/name).resolve()
 if not any(f.is_relative_to((parent/folder).resolve()) for folder in ['reference_history','candidate_history']) or not f.name.lower().startswith(a.variant.lower()) or not allowed(f):
  raise SystemExit('Variant archive must be a model-prefixed source under this benchmark reference_history/candidate_history: '+str(f))
 archives.append({'path':str(f.relative_to(root)),'sha256':verifier.hashlib.sha256(f.read_bytes()).hexdigest(),'scope':'Retained rejected variant source; not executed source or passing candidate.'});paths.append(f)
if archives:
 for owned_child in scoped_children:
  receipt_path=owned_child/'verification.json';receipt=json.loads(receipt_path.read_text());receipt['retained_rejected_variant_archives']=archives;receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
conditional=any(c.get('native_reference_repaired') for _,c,_ in data)
if conditional:
 if any(c.get('native_reference_repaired') and not c.get('reference_provenance') for _,c,_ in data):raise SystemExit('Corrected native references require explicit provenance.')
 title=('Repair '+a.benchmark+' native study reference with preserved witnesses') if a.shared else ('Conditionally verify '+a.benchmark+' '+a.variant+' against corrected native reference')
known_defect=any(c.get('native_behavior_known_scientific_defect') for _,c,_ in data)
if known_defect and conditional:raise SystemExit('Known-defect reproduction and corrected-native evidence need separate commit strata.')
if known_defect and any(c.get('native_behavior_known_scientific_defect') and not c.get('reference_provenance') for _,c,_ in data):raise SystemExit('Known scientific defects require witness provenance.')
if not paths:raise SystemExit('No owned files to commit')
subprocess.run(['git','add','-f','--',*[str(f.relative_to(root)) for f in paths]],cwd=root,check=True)
if known_defect:
 title=('Record '+a.benchmark+' original native behavior with known scientific defect') if a.shared else ('Verify '+a.benchmark+' '+a.variant+' native behavior with known scientific defect')
body=('Reproduced original native behavior with an independently witnessed scientific defect; not certification of intended algorithm, excluded from scientific-fidelity aggregates.\n\n' if known_defect else '')+('Conditional corrected-native stratum; no untouched-native agreement or aggregate claim.\n\n' if conditional else '')+'Evidence: '+', '.join(d['run_id'] for d,_,_ in data)+'. All five configured output checks pass; time means exclude trial1. Preserve numerical evidence strength, scope, attempt accounting and limitations.\n\nGenerated-By: OpenAI Codex'
subprocess.run(['git','commit','-m',title,'-m',body],cwd=root,check=True);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if key:
 m=root/'.porting-state/porting_logs'/(a.benchmark+'-julia-'+key)/'manifest.json'
 if m.exists():
  manifest=json.loads(m.read_text());
  if a.performance_tuning:
   manifest.setdefault('original_translation_commit',manifest.get('source_commit'));manifest.setdefault('performance_tuning_commits',[]).append({'commit':sha,'run_ids':[d['run_id'] for d,_,_ in data]});manifest.setdefault('performance_tuning_results',[]).extend({'configuration_id':c['configuration_id'],'run_id':d['run_id'],'commit':sha,'five_trial_seconds':[t['time_seconds'] for t in d['models'][key]['trials']],'mean_last_four_seconds':d['models'][key]['mean_seconds'],'attempt_accounting':c.get('attempt_accounting'),'token_accounting':c.get('token_accounting')} for d,c,_ in data);manifest['original_translation_fields_scope']='Original candidate/compiler/token and top-level timing fields retain initial translation evidence; later resourcepolicy results are separate performance_tuning_results.'
  if a.measurement_repair:
   manifest.setdefault('original_translation_commit',manifest.get('source_commit'))
   manifest.setdefault('measurement_repair_commits',[]).append({'commit':sha,'run_ids':[d['run_id'] for d,_,_ in data],'scope':'Completion/ordering protocol correction, not a new translation or performance tuning.'})
   manifest.setdefault('measurement_repair_results',[]).extend({'configuration_id':c['configuration_id'],'run_id':d['run_id'],'commit':sha,'five_trial_seconds':[t['time_seconds'] for t in d['models'][key]['trials']],'mean_last_four_seconds':d['models'][key]['mean_seconds']} for d,c,_ in data)
  if a.evidence_refresh:
   manifest.setdefault('verification_commits',[]).append({'commit':sha,'configurations':[c['configuration_id'] for _,c,_ in data],'note':'Evidence-link refresh only; source_commit retains actual unchanged implementation commit.'})
  else:manifest['source_commit']=sha
  manifest['commit_status']='committed';temp=m.with_suffix('.json.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(m)
subprocess.run([sys.executable,str(root/'tools/checkpoint_portable_study.py')],cwd=root,check=True)
print(json.dumps({'benchmark':a.benchmark,'phase':'shared' if a.shared else a.variant,'commit':sha}))
