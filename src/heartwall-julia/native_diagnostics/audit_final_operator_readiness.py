from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[3];P=R/'src/heartwall-julia';D=R/'.porting-state/diagnostics/heartwall/raw_capture';layout=json.load(open(P/'buffer_layout.json'))['buffers'];ref=np.memmap(P/'oracle_final_f64.bin',dtype='<f8',mode='r');budget=np.memmap(P/'oracle_final_budget_f64.bin',dtype='<f8',mode='r');records=[]
for b in layout:
 p=D/f'frame10_{b["name"]}.bin'
 if not p.exists():continue
 a=np.fromfile(p,dtype='<i4'if b['type']=='int32'else'<f4').astype(np.float64);assert len(a)==b['size'];off=b['final_offset']+16;x=ref[off:off+len(a)];e=budget[off:off+len(a)];valid=e>=0;diff=np.abs(a-x);records.append({'field':b['name'],'full_values':len(a),'direct_array_oracle_values':int(valid.sum()),'bad':int(np.count_nonzero(diff[valid]>e[valid])),'max_abs_error':float(np.max(diff[valid]))if np.any(valid)else None,'source_capture':'ActualpostkernelrawD2H, noguardsinvented','dynamic_operator_check_required':b['oracle_kind']=='actual_map_exact_argmax'})
x={'benchmark':'heartwall','scope':'Readinessactualunchangedsourceframe10fullcapturedfields, notprimarytrialdata','all_static_array_operator_budgets_pass':all(x['bad']==0 for x in records),'arrays':records,'final_operator_source':'src/heartwall-julia/generate_final_operator_oracle.py','clinical_accuracy_claim':False};(P/'native_diagnostics/final_operator_readiness_audit.json').write_text(json.dumps(x,indent=2)+'\n');print('PASS',x['all_static_array_operator_budgets_pass']);print([r for r in records if r['bad']])
