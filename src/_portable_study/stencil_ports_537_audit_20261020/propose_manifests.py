"""Read-only source evidence join; writes proposals only to requested private output.
NEVER writes shared manifests or source/receipt evidence. Root reviews proposals.
"""
from pathlib import Path
import json,hashlib,argparse,collections,sys
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--census',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.source_root.resolve();T=R/'.porting-state/portable_julia';out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);rows={x['benchmark']:x for x in json.load(open(a.census))['rows']};owned=json.load(open(T/'workers/stencil_ports_expansion537.json'))['completed'];reviews=[]
def read(f):return json.load(open(f))if f.exists()else{}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def alias(d,names):return next(((d[k],k)for k in names if d.get(k)is not None),(None,None))
for b in owned:
 for m in ['cuda_jl','ka','jacc']:
  P=R/'src'/(b+'-julia');child=P/(b+'-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[m]);main=child/'main.jl';target=R/f'.porting-state/porting_logs/{b}-julia-{m}/manifest.json';old=read(target);docs=[(target,old)]if old else[]
  for al in {'cuda_jl':['cuda.jl','cuda'],'ka':['KA'],'jacc':['JACC']}[m]:
   f=R/f'.porting-state/porting_logs/{b}-julia-{al}/manifest.json'
   if f.exists():docs.append((f,read(f)))
  for f in [child/'translation_accounting.json',child/'study_manifest.json',P/'effort_summary.json']:
   if f.exists():docs.append((f,read(f)))
  row={'benchmark':b,'model':m,'destination':str(target.relative_to(R)),'original_sha256':sha(target)if target.exists()else None,'fields_added':{},'warnings':[]}
  if b not in rows or not main.exists():row['status']='missing_evidence';reviews.append(row);continue
  run=R/rows[b]['result'];d=read(run);rec=d['models'][m];h=sha(main);bound=rec.get('adapter_file_hashes',{}).get(str(main));bound=bound.get('sha256')if isinstance(bound,dict)else bound
  if bound!=h:
   sys.path.insert(0,str(R/'tools'));from verify_portable_julia import digest_tree
   if digest_tree(rec.get('source_path','/nonexistent'))!=rec.get('source_hash'):row['status']='source_binding_mismatch';reviews.append(row);continue
  new=dict(old)
  def add(k,v,e):
   if v is None:return
   if k not in new or new[k]is None:new[k]=v;row['fields_added'][k]={'value':v,'evidence':e}
   elif new[k]!=v:row['warnings'].append({'existing_preserved':k,'existing':new[k],'alternative':v,'evidence':e})
  add('benchmark_id',b,'Ownedqueue');add('implementation_model',m,'Ownedqueue');add('source_sha256',h,str(run.relative_to(R))+'actualmainbinding');add('verification_result',str(run.relative_to(R)),'Selectedall4savedreceipt');add('configuration_id',d['configuration_id'],'Selectedsavedreceipt');add('status','verified','All4namedcontract; qualificationremainsexplicitinconfig')
  fields={'candidate_attempts':['candidate_attempts','translation_attempts','iteration_count'],'candidate_attempts_unit':['candidate_attempts_unit','candidate_unit','candidate_attempts_definition','attempt_unit'],'compiler_check_invocations':['compiler_check_invocations','compiler_checks','compiler_loop_attempts','actual_complete_program_checks'],'actual_device_jit_process_checks':['actual_device_jit_process_checks','actual_device_JIT_candidate_checks','authored_device_jit_processes','authored_device_JIT_check_programs','actual_device_JIT_programs','actual_device_JIT_processes','actual_translated_device_JIT_processes'],'compiler_feedback_repairs':['compiler_feedback_repairs']}
  for key,aliases in fields.items():
   for f,doc in docs:
    v,aliasname=alias(doc,aliases)
    if v is not None:add(key,v,str(f.relative_to(R))+':'+aliasname);break
  eff=read(P/'effort_summary.json');units=eff.get('candidate_units',{}).get(m,{})
  # Explicit nested source records retain their original units and phase.
  for group in ['attempt_accounting','attempts']:
   entry=eff.get(group,{}).get(m,{})
   if entry:docs.append((P/'effort_summary.json',entry))
  if isinstance(eff.get(m),dict):docs.append((P/'effort_summary.json',eff[m]))
  for key,aliases in fields.items():
   for f,doc in docs:
    v,n=alias(doc,aliases)
    if v is not None:add(key,v,str(f.relative_to(R))+':explicitnested:'+n);break
  for field,groups in [('candidate_attempts',['per_model_complete_source_candidates','target_candidate_counts','per_model_complete_candidates']),('actual_device_jit_process_checks',['per_model_actual_device_JIT_processes','per_model_actual_device_JIT_programs'])]:
   for group in groups:
    value=eff.get(group,{}).get(m)
    if type(value)is int:add(field,value,'effort_summary.json:'+group+'.'+m);break
  text=eff.get('candidate_unit')
  if text:
   add('candidate_attempts_unit',text,'effort_summary.json:candidate_unit')
   if 'one actual device-JITprogramprocess pertarget'in text:add('actual_device_jit_process_checks',1,'Explicitwrittenone-per-targetcount:'+text)
  for group in ['attempts']:
   entry=eff.get(group,{}).get(m,{})
   value,key=alias(entry,['complete_program_submissions'])
   if value is not None:add('compiler_check_invocations',value,'Recordedwholeprogramsubmissions; notbackendcompiler count')
   value,key=alias(entry,['actual_device_JIT_programs','actual_translated_device_JIT_processes'])
   if value is not None:add('actual_device_jit_process_checks',value,'effort_summary.json:attempts.'+m+'.'+key)
   unit,key=alias(entry,['unit','attempt_unit'])
   if unit:add('candidate_attempts_unit',unit,'effort_summary.json:attempts.'+m+'.'+key)
  if eff.get('effort_units',{}).get('per_model_complete_candidates',{}).get(m)is not None:
   add('candidate_attempts',eff['effort_units']['per_model_complete_candidates'][m],'effort_summary.json:effort_units.per_model_complete_candidates.'+m)
   add('candidate_attempts_unit','Completeprogramcandidates only; earlier29sourcegenerator/25subsystemJIT processes separate, not totalworkflowcompilerinvocations.','effort_summary.json:effort_units')
  for key,aliases in {'candidate_attempts':['whole_program_evaluations'],'compiler_check_invocations':['whole_program_evaluations'],'actual_device_jit_process_checks':['actual_device_JIT_processes','actual_device_JIT_process_checks']}.items():
   v,n=alias(units,aliases)
   if v is not None:add(key,v,'effort_summary.json candidate_units.'+m+'.'+n)
  if units.get('whole_program_evaluations')is not None:add('compiler_checks_unit','Recorded wholeprogram/frontend/host evaluation; backend internalcompiler invocations unknown.','Explicitcandidateunit')
  boundary=None
  for f,doc in docs:
   u=doc.get('usage',{});path=doc.get('usage_boundary_file')or(u.get('boundary_file')if isinstance(u,dict)else None)
   if path:boundary=R/path;break
  if boundary is None:boundary=next((f for phase in['translation_feedback','translation']if(f:=T/f'usage_boundaries_300/{b}/{m}_{phase}.jsonl').exists()),None)
  if boundary and boundary.exists():
   ends=[json.loads(line)for line in boundary.read_text().splitlines()if json.loads(line).get('boundary')=='end']
   if ends:
    z=ends[-1];existing=new.get('translation_tokens');delta=z.get('delta_usage')
    if isinstance(existing,dict)and isinstance(delta,dict):
     for key,value in delta.items():
      if key not in existing:existing[key]=value;row['fields_added']['translation_tokens.'+key]={'value':value,'evidence':str(boundary.relative_to(R))+':'+z['event_id']}
      elif existing[key]!=value:row['warnings'].append({'existing_token_preserved':key,'existing':existing[key],'alternative':value})
    else:add('translation_tokens',delta,str(boundary.relative_to(R))+':'+z['event_id']);add('usage_boundary_file',str(boundary.relative_to(R)),'Originalclosedwindow');add('usage_begin_event_id',z.get('begin_event_id'),'Originalevent');add('usage_end_event_id',z['event_id'],'Originalevent');add('token_accounting_scope','Inclusiveinput/outputphase counters; cachedinputsubsetinput/reasoningsubsetoutput; orderedcontextreuse, notexclusivebilling/intrinsiccoldstartcost.','Originalboundaryscope')
  if not new.get('candidate_attempts_unit'):row['warnings'].append('Candidateunitnotrecordedbydelivery: remainsunknown')
  new['normalization_review']={'kind':'Proposedadditivealiasesonly; originalfields/rootcommitmetadata preserved. Sharedmanifest notwrittenbyworker.','source_evidence':[{'path':str(f.relative_to(R)),'sha256':sha(f)}for f,_ in docs if f.exists()and f!=target],'selected_result':str(run.relative_to(R)),'selected_result_sha256':sha(run),'current_main_sha256':h}
  dest=out/(b+'__'+m+'.json');dest.write_text(json.dumps(new,indent=2)+'\n');row.update(status='proposal_only',proposal=str(dest),proposal_sha256=sha(dest));reviews.append(row)
(out/'review.json').write_text(json.dumps({'scope':'Read-onlyoriginalevidence; proposalJSONsonly, no sharedwrites. Rootmustreview/concurrencycheckoriginalSHA beforeanyapplication.','script_sha256':sha(Path(__file__)),'rows':reviews},indent=2)+'\n');print(json.dumps({'records':len(reviews),'status':dict(collections.Counter(x['status']for x in reviews)),'with_changes':sum(bool(x['fields_added'])for x in reviews),'warnings':sum(len(x['warnings'])for x in reviews)}))
