#!/usr/bin/env python3
"""ROOT-REVIEWED future CPU publication export recipe. Authored, never run here.
No summarize, profiling, GPU, signal, source mutation, install or build operation.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,subprocess,csv
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/imo/HeCBench_julia')
OUT=Path('/tmp/roofline-paper-coherent-20261022-r2')
LABEL='22 October 2026; partial profiling campaign'
STUDY=ROOT/'ontology/studies/v100_roofline_537'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def commands():
 py='/usr/bin/python3';base=['--source-root',str(ROOT)]
 exports=[
 [py,str(ROOT/'tools/export_v100_retained_roi.py'),*base,'--output-root',str(OUT/'family')],
 [py,str(ROOT/'tools/export_v100_benchmark_assessment.py'),*base,'--output-root',str(OUT/'assessment'),'--retained-family-audit',str(OUT/'family/audit.json')],
 [py,str(ROOT/'tools/export_v100_measurement_failures.py'),*base,'--output-root',str(OUT/'failures')]]
 detached=[
 [py,str(ROOT/'2026-PASC-Hecbench-Julia-paper/tools/prepare_v100_failure_tables.py'),'--snapshot',str(OUT/'failures'),'--output-root',str(OUT/'failure_tables'),'--snapshot-label',LABEL],
 ['/home/imo/noback/conda/bin/python3.9',str(ROOT/'tools/plot_v100_benchmark_assessment.py'),*base,'--descriptor',str(OUT/'assessment/descriptor_assessment.json'),'--family-audit',str(OUT/'family/audit.json'),'--output-root',str(OUT/'figures'),'--snapshot-label',LABEL]]
 return exports,detached

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-reviewed',action='store_true');a=p.parse_args();exports,detached=commands()
 if not a.execute_reviewed:
  print(json.dumps({'not_executed':True,'root_prerequisites':'source13-15 archive/adoption + postannotation success and matching current frozen audit; root confirms explicit22Oct label','one_shared_lock_group':exports,'then_detached':detached},indent=2));return
 pins=load(HERE/'tool_bindings.json')
 for path,h in pins.items():
  if sha(path)!=h:raise RuntimeError('Tool/source binding drift; review again: '+path)
 if OUT.exists()or OUT.is_symlink():raise RuntimeError('Fresh output root required')
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  progress=load(STUDY/'PROGRESS.json');audit=load(STUDY/'integrity_audit.json')
  if progress['benchmark_population']!=537 or progress['model_slots']!=2148:raise RuntimeError('Canonical denominator changed')
  if audit['issues']or audit['attempts']!=progress['retained_profile_attempts']:raise RuntimeError('Current progress and complete frozen audit differ; wait coordinator pipeline, do not merge generations')
  audited={x['profile_id']for x in audit['records']if x.get('audit_status')=='pass'}
  for lp in (STUDY/'benchmarks').glob('*.json'):
   for m,s in load(lp).get('models',{}).items():
    mp=s.get('profile_evidence_manifest')
    if mp and str(Path(mp).parent)not in audited:raise RuntimeError('Selected capture absent from matching frozen audit: '+str(mp))
  core=[STUDY/'PROGRESS.json',STUDY/'integrity_audit.json',STUDY/'roi_reviews/INDEX.json'];corepins={str(p):sha(p)for p in core};OUT.mkdir(mode=0o700)
  write(OUT/'admission.json',{'scope':'Dated partial-campaign snapshot; not new measurement','label':LABEL,'core_input_bindings':corepins,'tool_bindings':pins,'progress':progress,'audit_attempts':audit['attempts'],'audit_issues':audit['issues'],'source_adoption_required_external_root_receipt':True,'whole_ROI_claim':False,'causal_claim':False})
  for c in exports:subprocess.run(c,cwd=ROOT,check=True)
  if any(sha(p)!=h for p,h in corepins.items()):raise RuntimeError('Canonical input drift while SH locked')
  fcntl.flock(lock,fcntl.LOCK_UN)
 for c in detached:subprocess.run(c,cwd=ROOT,check=True)
 tables=load(OUT/'failure_tables/packet_summary.json')
 if tables['publication_ready']is not True:raise RuntimeError('Descriptive taxonomy not publication ready')
 figs=load(OUT/'figures/provenance.json')
 if figs['plotted_admitted_family_count']!=12 or figs['strict_family_missingness']:raise RuntimeError('Strict12 family evidence changed; separately review, never replace historical plot implicitly')
 # Preserve strict scientific totals rather than requiring obsolete metadata-byte hashes.
 oldcsv=ROOT/'2026-PASC-Hecbench-Julia-paper/data/v100_assessment/plotted_integrated_intensity.csv'
 fields=('benchmark','model','actual_scientific_launches','fp32_executed_flops','dram_bytes','integrated_fp32_flop_per_DRAM_byte')
 def scientific_rows(path):
  with path.open(newline='')as f:return sorted(tuple(r[k]for k in fields)for r in csv.DictReader(f))
 if scientific_rows(oldcsv)!=scientific_rows(OUT/'figures/plotted_integrated_intensity.csv'):raise RuntimeError('Strict12 scientific values differ from old plotted bytes; require separate review, do not silently replace')
 selected=tables['statistics']['selected']['per_model_group_counts']
 if sum(selected[m].get('completed_counter_observation',0)for m in ('cuda','cuda_jl','ka','jacc'))!=progress['status_counts'].get('profile_process_completed',0):raise RuntimeError('Selected census and canonical completed-process counts disagree; no pooled-generation publication')
 if any(tables['statistics']['selected']['per_model_records'][m]!=537 for m in ('cuda','cuda_jl','ka','jacc')):raise RuntimeError('Per-model failure-table denominator mismatch')
 macros={'VhundredSnapshotLabel':LABEL,'VhundredCompleted':progress['status_counts'].get('profile_process_completed',0),'VhundredAllFour':progress['names_with_all_four_completed_profiles'],'VhundredAttempts':progress['retained_profile_attempts'],'VhundredSourceNA':progress['status_counts'].get('roofline_not_applicable_no_device_kernels',0),'VhundredHardware':progress['status_counts'].get('native_hardware_blocked',0)}
 for model,name in [('cuda','Native'),('cuda_jl','CudaJl'),('ka','Ka'),('jacc','Jacc')]:macros['Vhundred'+name+'Completed']=selected[model].get('completed_counter_observation',0)
 # This uses actual frozen outcome counts, never forecasts or expected source adoption totals.
 (OUT/'paper_snapshot_values.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n'for k,v in macros.items()))
 write(OUT/'packet_manifest.json',{'label':LABEL,'root':str(OUT),'commands':exports+detached,'files':{str(p.relative_to(OUT)):sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()},'source_adoption_external_receipt_still_required':True,'no_paper_files_modified':True,'whole_ROI_claim':False,'causal_claim':False})
 print(json.dumps({'fresh_output':str(OUT),'publication_data_ready_for_root_review':True,'paper_not_changed':True}))
if __name__=='__main__':main()
