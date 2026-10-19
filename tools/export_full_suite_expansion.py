#!/usr/bin/env python3
"""Join prospective full-suite records without altering the frozen447 study."""
from pathlib import Path
import csv,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'.porting-state/portable_julia';PAPER=ROOT/'2026-PASC-Hecbench-Julia-paper'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def tex(s):return str(s).replace('_',r'\_').replace('&',r'\&')
def main():
 cohort=read(STATE/'batch_537.json');new=set(cohort['new_benchmarks']);latest={};partial={}
 for p in sorted((STATE/'runs').glob('*/result.json')):
  if p.parent.name.split('_',1)[-1] not in new:continue
  d=read(p);b=d.get('benchmark')
  if b not in new:continue
  c=read(p.with_name('config.json'),{})
  partial[b]=(d,c,p)
  if d.get('reference_agreement') and all(d.get('variant_verified',{}).get(m) for m in ('ka','jacc')):latest[b]=(d,c,p)
 rows=[]
 for b in sorted(new):
  record=latest.get(b) or partial.get(b);traits=read(STATE/'traits'/f'{b}.json',{});models={}
  row={'benchmark':b,'status':'unstarted_or_readiness_only','scientific_intent':traits.get('scientific_intent'),'models':models}
  if record:
   d,c,p=record;row.update(status='accepted_named_contract' if b in latest else 'in_progress_not_four_model_accepted',configuration_id=d['configuration_id'],result_path=str(p.relative_to(ROOT)),result_sha256=sha(p),protocol=d['protocol'],input_prefix_count=d.get('input_prefix_count'),reference_qualification='corrected_native_reference' if c.get('native_reference_repaired') else 'known_native_scientific_defect' if c.get('native_behavior_known_scientific_defect') else 'revised_contract' if c.get('verification_contract_revised') else 'ordinary_named_contract',timing_scope=c.get('timing_region'),reference_provenance=c.get('reference_provenance'),scientific_witness=read(STATE/'scientific_witnesses_300'/f'{b}.json'))
   for m in ('cuda','cuda_jl','ka','jacc'):
    v=d.get('models',{}).get(m,{});manifest=read(ROOT/f'.porting-state/porting_logs/{b}-julia-{m}/manifest.json',{})
    models[m]={'status':v.get('status'),'complete':v.get('complete'),'mean_last_four_seconds':v.get('mean_seconds'),'five_trial_seconds':[t.get('time_seconds') for t in v.get('trials',[])],'all_five_checks_pass':len(v.get('trials',[]))==5 and all(t.get('correct') for t in v['trials']),'checked_values_per_trial':[t.get('checked_values') for t in v.get('trials',[])],'source_hash':v.get('source_hash'),'adapter_file_hashes':v.get('adapter_file_hashes'),'candidate_attempts':manifest.get('candidate_attempts'),'compiler_checks':manifest.get('compiler_check_invocations'),'actual_device_jit_process_checks':manifest.get('actual_device_jit_process_checks'),'compiler_feedback_repairs':manifest.get('compiler_feedback_repairs'),'translation_tokens':manifest.get('translation_tokens'),'source_commit':manifest.get('source_commit'),'implementation_traits':traits.get('implementation_traits',{}).get(m)}
  rows.append(row)
 accepted=[r for r in rows if r['status']=='accepted_named_contract'];cal=read(PAPER/'research/expansion_537/token_cost_calibration.json',{})
 doc={'snapshot_created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_native_names':537,'frozen_primary_population':447,'new_frontier_names':90,'new_four_model_accepted_names':len(accepted),'new_not_yet_four_model_accepted_names':90-len(accepted),'frozen_primary_analysis_changed':False,'published_v1_0_0_changed':False,'frozen_primary_contracts':{'numerical':436,'property':2,'execution_only':5,'native_hardware_barrier':4},'expanded_accepted_numerical_contracts':436+len(accepted),'count_scope':'Qualified named numerical contracts, including explicit reference qualifications; not universal scientific validity or completion of537.','translation_order':'CUDAjl thenKA thenJACC within benchmark; shared helper and stream context inherited.','calibration':{'observed_windows':cal.get('observation_windows'),'eligible_windows':cal.get('eligible_windows'),'fitted_strata':sum(f.get('status')!='insufficient_calibration' for f in cal.get('fits',[])),'historical_actuals_modified':False,'scope':'Expected present-workflow forecasts separate from unknown historical actuals; insufficient per-stratum sample yields no estimate.'},'rows':rows}
 write(PAPER/'research/expansion_537/live_coverage.json',doc)
 compact={k:v for k,v in doc.items() if k!='rows'};compact['accepted_cases']=[{k:v for k,v in r.items() if k not in ('scientific_witness','reference_provenance')} for r in accepted]
 for r in compact['accepted_cases']:
  r['models']={m:{k:v for k,v in v.items() if k not in ('adapter_file_hashes','implementation_traits')} for m,v in r['models'].items()}
 write(PAPER/'data/prospective_extension.json',compact)
 out=PAPER/'sections/generated/full_suite_extension.tex';out.parent.mkdir(parents=True,exist_ok=True)
 lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Prospective extension: named kernel-region means in milliseconds (five checked trials, mean of last four). C: corrected native host; D: known native defect. These cases are outside the frozen447 analysis.}',r'\label{tab:prospective-extension}',r'\begin{tabular}{lrrrrl}',r'\toprule Case & CUDA & CUDA.jl & KA & JACC & Ref.\\',r'\midrule']
 for r in accepted[:8]:
  tag={'corrected_native_reference':'C','known_native_scientific_defect':'D','ordinary_named_contract':'ordinary','revised_contract':'revised'}[r['reference_qualification']]
  times=['{:.3f}'.format(1000*r['models'][m]['mean_last_four_seconds']) for m in ('cuda','cuda_jl','ka','jacc')]
  lines.append(tex(r['benchmark'])+' & '+' & '.join(times)+' & '+tag+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\newcommand{\extensionAccepted}{'+str(len(accepted))+'}',r'\newcommand{\extensionPending}{'+str(90-len(accepted))+'}'])
 out.write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:v for k,v in doc.items() if k in ['new_four_model_accepted_names','new_not_yet_four_model_accepted_names','expanded_accepted_numerical_contracts']}))
if __name__=='__main__':main()
