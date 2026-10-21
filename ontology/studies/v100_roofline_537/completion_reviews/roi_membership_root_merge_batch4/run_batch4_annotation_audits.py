from pathlib import Path
import subprocess,json,datetime
R=Path('/home/imo/HeCBench_julia');P=Path('/tmp/roofline-completion-20261021');base=['python3'];steps=[('summarize',[str(R/'tools/v100_roofline_summarize.py'),'--source-root',str(R),'--raw-root','/tmp/hecbench-roofline-v100-20261020']),('integrity',[str(R/'tools/v100_roofline_audit.py'),'--source-root',str(R),'--output',str(R/'ontology/studies/v100_roofline_537/integrity_audit.json')]),('postannotation',[str(P/'roi_batch4_postannotation_validator/root_validate_postannotation_callable.py'),'--source-root',str(R),'--approval-plan',str(P/'roi_batch4_root_dryrun_004/merge_plan.json'),'--apply-report',str(P/'roi_batch4_root_apply_001/validation_report.json'),'--strict-baseline',str(P/'roi_batch4_strict_family_baseline.json'),'--output-root',str(P/'roi_batch4_postannotation_actual_001')])]
for label,argv in steps:
 log=P/('roi_batch4_'+label+'.log')
 with log.open('xb')as out:p=subprocess.run(base+argv,cwd=R,stdout=out,stderr=subprocess.STDOUT)
 print(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':label,'returncode':p.returncode,'log':str(log)}),flush=True)
 if p.returncode:raise SystemExit(p.returncode)
print('All actual batch4 annotation audits passed.',flush=True)
