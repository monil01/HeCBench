#!/usr/bin/env python3
"""Coordinator commit helper: require dual-reference evidence and unchanged kernel source.
Does not run benchmarks. Shared reference work and each variant are separate commits.
"""
import argparse,json,math,subprocess,sys
from pathlib import Path
from study_evidence import effective_config
import verify_portable_julia as verifier
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--result',type=Path,action='append',required=True);p.add_argument('--variant',choices=['KA','JACC']);p.add_argument('--shared',action='store_true');p.add_argument('--reviewed-dependency-change',default='');p.add_argument('--performance-tuning',action='store_true');p.add_argument('--evidence-refresh',action='store_true');a=p.parse_args()
if bool(a.variant)==bool(a.shared):p.error('choose exactly one of --variant or --shared')
if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip():raise SystemExit('Index already contains staged work; inspect it before using coordinator helper.')
data=[]
for path in a.result:
 if not path.is_absolute():path=root/path
 d=json.loads(path.read_text());c=effective_config(path,json.loads(path.with_name('config.json').read_text()))
 if d['benchmark']!=a.benchmark or not d.get('reference_agreement'):raise SystemExit('Benchmark or reference agreement invalid')
 data.append((d,c,path))
parent=root/'src'/(a.benchmark+'-julia');key={'KA':'ka','JACC':'jacc'}.get(a.variant)
def allowed(f):return f.is_file() and (f.suffix in ['.jl','.cu','.cuh','.h','.hpp','.cpp','.toml','.md','.json','.sh'] or f.name in ['Makefile','original_Makefile'])
if a.shared:
 for d,c,path in data:
  for model in ['cuda','cuda_jl']:
   if not d['models'][model].get('complete'):raise SystemExit('Incomplete reference')
   if verifier.digest_tree(verifier.discover(a.benchmark)[model])!=d['models'][model]['source_hash']:raise SystemExit('Reference source changed since verification: '+model)
 paths=[f for f in parent.iterdir() if allowed(f)]+[f for f in (parent/(a.benchmark+'-julia-cuda.jl')).rglob('*') if allowed(f)]
 title='Align '+a.benchmark+' study fixtures and verified references'
else:
 child=parent/(a.benchmark+'-julia-'+a.variant);proof=[]
 for d,c,path in data:
  if not d['variant_verified'].get(key):raise SystemExit('Variant not verified against both references')
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
  proof.append({'native_reference_repaired':bool(c.get('native_reference_repaired')),'native_behavior_known_scientific_defect':bool(c.get('native_behavior_known_scientific_defect')),'reference_provenance':c.get('reference_provenance'),'configuration_classification_adjudication':c.get('configuration_classification_adjudication'),'untouched_native_aggregate_eligible':not c.get('native_reference_repaired') and not c.get('exclude_untouched_native_aggregates',False) and not c.get('exclude_from_scientific_fidelity_aggregates',False) and not c.get('native_behavior_known_scientific_defect',False),'run_id':d['run_id'],'configuration_id':c['configuration_id'],'protocol':d['protocol'],'variant_verified':True,'reference_agreement':True,'evidence_strength':c.get('numerical_evidence_strength') or ('aggregate_count_and_error_sum_only' if a.benchmark=='feynman-kac' else 'full_output_artifacts'),'comparisons':d['comparisons'][key],'models':d['models'],'ontology_context':d.get('ontology'),'input_contract':d.get('input_contract'),'attempt_accounting':d.get('attempt_accounting'),'token_accounting':d.get('token_accounting'),'reviewed_dependency_change':{'reason':a.reviewed_dependency_change,'paths':changed} if changed else None,'raw_result':str(path.relative_to(root))})
 child.joinpath('verification.json').write_text(json.dumps({'benchmark':a.benchmark,'variant':'julia-'+key,'configurations':proof},indent=2)+'\n');paths=[f for f in child.rglob('*') if allowed(f)];title=('Link verified ' if a.evidence_refresh else 'Tune verified ' if a.performance_tuning else 'Add verified ')+a.benchmark+' '+a.variant+(' primary evidence' if a.evidence_refresh else ' implementation')
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
  if a.evidence_refresh:
   manifest.setdefault('verification_commits',[]).append({'commit':sha,'configurations':[c['configuration_id'] for _,c,_ in data],'note':'Evidence-link refresh only; source_commit retains actual unchanged implementation commit.'})
  else:manifest['source_commit']=sha
  manifest['commit_status']='committed';temp=m.with_suffix('.json.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(m)
subprocess.run([sys.executable,str(root/'tools/checkpoint_portable_study.py')],cwd=root,check=True)
print(json.dumps({'benchmark':a.benchmark,'phase':'shared' if a.shared else a.variant,'commit':sha}))
