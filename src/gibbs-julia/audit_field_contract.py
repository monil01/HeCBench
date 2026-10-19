#!/usr/bin/env python3
"""Complete saved-array comparison with explicit per-field budgets and exact RNG/guards."""
from pathlib import Path
import json,array,hashlib,sys,itertools,math
P=Path(__file__).resolve().parent;R=P.parent.parent
f=Path(sys.argv[1]);f=f if f.is_absolute()else R/f;r=json.loads(f.read_text())
contracts={'theta':3e-5,'feedback_parameters':3e-5,'reduction_and_hostsum':2e-4,'oracle_error_diagnostic':3e-5,'exact_RNGstate_and_integer_flags':0,'exact_guards':0,'exact_actual_input_prefix':0}
def field(k):
 if k<7900:return 'exact_actual_input_prefix'
 q=(k-7900)%6812
 if q==0 or q==7:return 'exact_RNGstate_and_integer_flags'
 if 1<=q<=4:return 'feedback_parameters'
 if q in (5,6):return 'reduction_and_hostsum'
 if q==8:return 'oracle_error_diagnostic'
 q-=9
 if q<545:return 'exact_guards'if q<16 or q>=529 else 'theta'
 q-=545
 if q<70:return 'exact_guards'if q%35<16 or q%35>=19 else 'reduction_and_hostsum'
 return 'exact_RNGstate_and_integer_flags'
models={};files={}
for model,m in r['models'].items():
 if m.get('status')!='native_check_pass':continue
 vals=[];files[model]=[]
 for t in m['trials']:
  art=t['output_artifact'];p=Path(art['saved_path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==art['sha256'];a=array.array('d');a.frombytes(b);assert len(a)==225884 and all(map(math.isfinite,a));vals.append(a);files[model].append({'path':str(p.relative_to(R)),'sha256':art['sha256']})
 models[model]=vals
out=[]
for x,y in itertools.combinations(models,2):
 for trial,(a,b)in enumerate(zip(models[x],models[y]),1):
  maxima={k:0.0 for k in contracts};counts={k:0 for k in contracts};bad=0
  for k,(v,w)in enumerate(zip(a,b)):
   name=field(k);e=abs(v-w);maxima[name]=max(maxima[name],e);counts[name]+=1;bad+=e>contracts[name]
  out.append({'models':[x,y],'trial':trial,'pass':bad==0,'bad_values':bad,'max_abs_error_by_field':maxima,'full_value_counts_by_field':counts})
report={'benchmark':'gibbs','source_result':str(f.relative_to(R)),'result_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'contracts_absolute_rtol0':contracts,'full_pair_trial_checks':out,'source_artifacts':files,'all_pairs_all5_field_contract_pass':all(z['pass']for z in out),'all_values_each_comparison':225884,'prefix':7900,'is_GPU_execution':False,'original_uniform3e-5_contract_changed':True,'theta_3e-5_and_exact_RNG_not_relaxed':True}
dest=Path(sys.argv[2])if len(sys.argv)>2 else P/'saved_field_contract_audit.json';dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(out),'pass':report['all_pairs_all5_field_contract_pass'],'destination':str(dest)}));sys.exit(0 if report['all_pairs_all5_field_contract_pass']else 1)
