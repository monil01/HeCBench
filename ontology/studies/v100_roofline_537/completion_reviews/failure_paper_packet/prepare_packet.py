#!/usr/bin/env python3
"""Reproduce descriptive paper tables from a frozen failure census; CPU files only."""
import argparse,collections,csv,hashlib,json,pathlib
P=pathlib.Path;M=('cuda','cuda_jl','ka','jacc');LABEL={'cuda':'CUDA','cuda_jl':'CUDA.jl','ka':'KA','jacc':'JACC'}
CODES=[('completed_counter_observation','Retained counter record'),('foreign_gpu_contention','Foreign GPU stop'),('operational_wall_timeout','Operational timeout'),('build_or_helper_diagnostic','Build/helper/I/O/checker diagnostic'),('profile_process_nonzero_cause_unknown','Nonzero process; cause unknown'),('csv_field_size_limit','CSV field-size limit'),('pending_or_active_not_completed','Pending/active'),('plan_isolation_or_device_gate','Plan/isolation gate'),('native_hardware_barrier','Hardware prerequisite'),('replay_method_hold','Replay-method hold'),('source_reviewed_runtime_no_kernel_not_applicable','Source-reviewed kernel roofline N/A')]
GROUP={'build_launcher_failure','compiler_child_nonzero_diagnostic','dependency_or_loader_diagnostic','filesystem_permission_diagnostic','output_io_diagnostic','assertion_wrapper_rejection_diagnostic'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rows,fields):
 with p.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def tex_escape(s):return s.replace('&',r'\&').replace('_',r'\_')
def main():
 a=argparse.ArgumentParser();a.add_argument('--snapshot',type=P,required=True);a.add_argument('--output-root',type=P,required=True);q=a.parse_args();src=q.snapshot.resolve();out=q.output_root.resolve()
 if out.exists():raise RuntimeError('Fresh output directory required')
 if not out.is_relative_to('/tmp')or out==P('/tmp'):raise RuntimeError('/tmp only')
 summary=json.loads((src/'summary.json').read_text());attempts=json.loads((src/'all_frozen_attempts.json').read_text())['records'];views=json.loads((src/'selected_latest_slots.json').read_text())['records']
 if not summary['integrity']['valid']or summary['integrity']['issues']or summary.get('live_ledger_drift_paths'):raise RuntimeError('Frozen source integrity/drift report not clean')
 bindings={str(p):sha(p)for p in [src/'summary.json',src/'all_frozen_attempts.json',src/'selected_latest_slots.json',src/'process_returncodes.csv',src/'exporter_source.py',P(__file__)]}
 out.mkdir();details=[];compact=[];stats={}
 for view in ['selected','latest','all_frozen_attempts']:
  rows=attempts if view=='all_frozen_attempts'else[x for x in views if x['view']==view];grouped={m:collections.Counter()for m in M};raw={m:collections.Counter()for m in M}
  for row in rows:
   category=row['operational_category'];model=row['model'];raw[model][category]+=1;grouped[model]['build_or_helper_diagnostic'if category in GROUP else category]+=1
  for model in M:
   denominator=sum(raw[model].values())
   if view!='all_frozen_attempts'and denominator!=537:raise RuntimeError('537-slot denominator')
   for cat,n in sorted(raw[model].items()):details.append({'view':view,'model':model,'category':cat,'count':n,'row_unit':'frozen_attempt_record'if view=='all_frozen_attempts'else'benchmark_model_slot','view_records':denominator,'canonical_name_denominator':537,'percentage':None})
  for cat,label in CODES:
   compact.append({'view':view,'category':cat,'label':label,**{m:grouped[m][cat]for m in M},'total':sum(grouped[m][cat]for m in M),'row_unit':'frozen_attempt_record'if view=='all_frozen_attempts'else'benchmark_model_slot'})
  stats[view]={'per_model_records':{m:sum(raw[m].values())for m in M},'per_model_group_counts':{m:dict(grouped[m])for m in M},'total_records':len(rows)}
  title='Selected model-slot outcomes'if view=='selected'else'Latest frozen model-slot outcomes'if view=='latest'else'All frozen attempt-record outcomes'
  lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{'+title+r' at the frozen 21 October 2026, 18:38 UTC snapshot. '+(r'Each model retains all 537 names.'if view!='all_frozen_attempts'else r'Repeated and CPU-derivative records are included; these are not independent benchmark counts.')+r'}',r'\label{tab:roofline-failures-'+view.replace('_','-')+r'}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Outcome & CUDA & CUDA.jl & KA & JACC \\',r'\midrule']
  for cat,label in CODES:
   if not any(grouped[m][cat]for m in M):continue
   lines.append(tex_escape(label)+' & '+' & '.join(str(grouped[m][cat])for m in M)+r' \\')
  lines.extend([r'\midrule','Records & '+' & '.join(str(sum(raw[m].values()))for m in M)+r' \\',r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
  (out/(view+'_table.tex')).write_text('\n'.join(lines))
 table(out/'taxonomy_detailed.csv',details,['view','model','category','count','row_unit','view_records','canonical_name_denominator','percentage'])
 table(out/'taxonomy_compact.csv',compact,['view','category','label',*M,'total','row_unit'])
 bysel={(x['benchmark'],x['model']):x for x in views if x['view']=='selected'};bylat={(x['benchmark'],x['model']):x for x in views if x['view']=='latest'};diff=[]
 for key,row in sorted(bysel.items()):
  other=bylat[key]
  if row['operational_category']!=other['operational_category']:
   diff.append({'benchmark':key[0],'model':key[1],'selected_category':row['operational_category'],'latest_category':other['operational_category'],'selected_profile_ids':json.dumps(row['profile_ids']),'latest_profile_ids':json.dumps(other['profile_ids'])})
 table(out/'selected_latest_differences.csv',diff,['benchmark','model','selected_category','latest_category','selected_profile_ids','latest_profile_ids'])
 processes=[]
 for m in M:
  model=next(x for x in summary['per_model']if x['model']==m);rr=[x for x in attempts if x['model']==m]
  processes.append({'model':m,'frozen_attempt_records':len(rr),'recorded_new_profiler_launchers':model['new_recorded_profile_launcher_processes'],'recorded_new_build_launchers':model['new_recorded_build_launcher_processes'],'CPU_derivative_records_new_execution_zero':model['CPU_recovery_records_no_new_execution'],'raw_profiler_returncode_null_records':sum(x.get('raw_profile_returncode')is None for x in rr),'underlying_device_JIT_invocations':None,'underlying_target_compiler_child_total':None,'historical_translation_attempts_inferred':None})
 table(out/'process_units_and_unknowns.csv',processes,list(processes[0]))
 codebook=[]
 for cat in sorted({x['category']for x in details}):
  codebook.append({'code':cat,'compact_group':'build_or_helper_diagnostic'if cat in GROUP else cat,'scope':'operational profiling outcome, not translation failure','inference':'retained log observation; unique cause unproved'if cat in GROUP else 'recorded process stop/outcome or explicit source assessment; no abstraction-quality inference','mutually_exclusive_priority':'exporter precedence; additional overlapping diagnostic signatures remain in all_frozen_attempts.json observations','missingness':'Absence of a category is not proof that its underlying event never happened outside retained history.'})
 dump(out/'codebook.json',codebook)
 payload={'snapshot_created_at':summary['created_at'],'source_progress_updated_at':summary['source_progress_updated_at'],'lock_acquired_at':summary.get('lock_acquired_at'),'lock_released_at':summary.get('lock_released_at'),'input_bindings':bindings,'source_integrity':{'issues':0,'qualifications':len(summary['integrity']['qualifications']),'raw_bytes_verified':summary['integrity']['all_manifest_uncompressed_bytes_verified']},'statistics':stats,'selected_vs_latest_different_categories':len(diff),'process_units':processes,'classification_note':'Mutually exclusive exporter priority; diagnostic group membership is descriptive, not independently established root cause. Raw labels and exact input paths/SHA retained.','token_note':'No token totals computed. Shared usage markers for census generation and paper support are separate windows with inherited/reused context; cached/reasoning fields are subsets and cannot be added. These windows do not measure translation cost.'}
 dump(out/'packet_summary.json',payload);print(json.dumps({'output':str(out),'selected_slots':stats['selected']['total_records'],'historical_records':stats['all_frozen_attempts']['total_records'],'selected_vs_latest_differences':len(diff)}))
if __name__=='__main__':main()
