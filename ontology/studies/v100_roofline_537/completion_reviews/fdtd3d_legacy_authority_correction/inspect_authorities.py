from pathlib import Path
import json,gzip,hashlib,collections,fcntl,datetime
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';bindings={}
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):
 p=Path(p);b=p.read_bytes();bindings[str(p)]=sha(b);return json.loads(gzip.decompress(b)if p.suffix=='.gz'else b),b
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 legacy,packed=load(S/'ROI_REVIEW.json.gz');receipt,_=load(S/'ROI_REVIEW_RECEIPT.json');need=sha(packed)==receipt['sha256']and sha(gzip.decompress(packed))==receipt['uncompressed_sha256']
 if not need:raise ValueError('Legacyreceiptmismatch')
 index,_=load(S/'roi_reviews/INDEX.json');keys=collections.Counter((x['benchmark'],x['model'])for x in legacy['records']);candidate=[]
 for row in index['cohorts']:
  doc,b=load(R/row['path'])
  if sha(b)!=row['compressed_sha256']or sha(gzip.decompress(b))!=row['raw_sha256']:raise ValueError('INDEXmismatch')
  candidate+=doc['records'];keys.update((x['benchmark'],x['model'])for x in doc['records'])
 loader=R/'tools/v100_roofline_summarize.py';loaderb=loader.read_bytes();bindings[str(loader)]=sha(loaderb);lines=loaderb.decode().splitlines();loaderproof={'path':str(loader),'sha256':sha(loaderb),'start_line':87,'end_line':104,'exact_lines':'\n'.join(lines[86:104])}
 ledger,_=load(S/'benchmarks/fdtd3d.json');modelrows=[]
 for m in ['cuda','cuda_jl','ka','jacc']:
  row=next(x for x in legacy['records']if x['benchmark']=='fdtd3d'and x['model']==m);state=ledger['models'][m];folder=(R/state['kernel_observations']['path']).parent;plan,pb=load(folder/'plan.json.gz');prov,vb=load(folder/'profile_provenance.json.gz');derived,db=load(folder/'derived.json.gz');annotated,ab=load(folder/'annotated.json.gz');captured={str(Path(p).resolve()):h for p,h in plan['execution_bindings'].items()};current_sources=[]
  for x in row['execution_bindings']:
   p=Path(x['path']).resolve()
   if str(p)in captured:
    raw=p.read_bytes();bindings[str(p)]=sha(raw)
    if sha(raw)!=captured[str(p)]or sha(raw)!=x['expected_sha256']:raise ValueError('Sourcejoin mismatch')
    current_sources.append({'path':str(p),'sha256':sha(raw)})
  if row['configuration_id']!=plan['configuration_id']or row['reference_result']['sha256']!=plan['reference_result_sha256']or row['raw_config_sha256']!=plan['raw_config_sha256']:raise ValueError('Legacy/currentcontractmismatch')
  if prov['status']!='profile_process_completed'or prov['profile']['returncode']!=0 or prov['profile']['stop_reason']is not None:raise ValueError('Selectednotcompleted')
  if prov['plan_sha256']!=sha(gzip.decompress(pb))or state['profile_provenance_sha256']!=sha(gzip.decompress(vb)):raise ValueError('Selectedjoin')
  modelrows.append({'model':m,'legacy_stages':row['stages'],'legacy_source_ROI':row['source_ROI'],'legacy_full_ROI_coverage':row['full_ROI_coverage'],'legacy_ordered_census':row['ordered_complete_launch_census_obtained'],'legacy_kernel_mappings':row['kernel_mappings'],'legacy_derived':row['derived'],'current_selected_profile':str(folder.relative_to(R)),'current_derived_raw_sha256':sha(gzip.decompress(db)),'same_derived_as_legacy':row['derived']['sha256']==sha(gzip.decompress(db)),'current_annotations':[{'identity':x['identity'],'ROI_annotation':x.get('ROI_annotation'),'sample_in_primary_timed_invocation':x.get('scope_context',{}).get('sample_in_primary_timed_invocation'),'retained_trial':x.get('scope_context',{}).get('retained_trial'),'whole_elapsed_ROI_claim':x.get('scope_context',{}).get('whole_elapsed_ROI_claim'),'causal_bottleneck':x.get('causal_bottleneck')}for x in annotated['records']],'legacy_current_source_joins':current_sources})
 for p,h in bindings.items():
  if sha(Path(p).read_bytes())!=h:raise ValueError('Changedinputduringreview')
 fcntl.flock(lock,fcntl.LOCK_UN)
report={'status':'CORRECTIVE_AUTHORITY_RECEIPT_NO_NEW_COHORT_OR_ANNOTATION_NEEDED','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supersedes':{'delivery_path':'/tmp/roofline-completion-20261021/fdtd3d_roi_record_genesis_review/DELIVERY.json','delivery_sha256':'44faaa2ee37589b8c967671cbabd9ba5a8411bb7cbed3b3fda0d843999d84672','proposal_sha256':'3c515d9ea47912c5b521dfe3eaf73c701484791411b3156e7fb05da73b49463a','qualification':'Its indexedcohortabsence/source/timer observations remain true, but proposedNEWauthority assumption is superseded: legacy canonicalauthority alreadycontains4fdtdrecords. Neverapplygenesisproposal.'},'legacy_receipt_verified':True,'loader_proof':loaderproof,'population':{'legacy_names':len({x['benchmark']for x in legacy['records']}),'legacy_model_records':len(legacy['records']),'indexed_candidate_names':len({x['benchmark']for x in candidate}),'indexed_candidate_model_records':len(candidate),'union_names':len({b for b,m in keys}),'union_models':len(keys),'duplicates':[{'benchmark':b,'model':m,'authorities':n}for(b,m),n in keys.items()if n>1]},'fdtd3d':modelrows,'user_facing_qualification':'fdtd3d already has four legacy source-family review records outside the indexed candidate cohorts. It adds zero new names/model records/declarations. The authority inventory combines9legacy names/36records and523candidate names/2092records; this532name/2128record presence does NOT mean532 source-closure proofs or completeROI captures. Currentfdtdsamples remainpartial; actualtimedphase/fullROI/causalproof not established.','existing_legacy_claim_to_qualify':'source_ROI.first_sample and legacykernelmapping warmup_source_order_inferred refer oldbatch001 and source-order inference, not runtimephaseproof for selectedbatch003. Currentselectedderiveddiffers; preserveoldrecords but do nottransferoldPIDs/exactmapping/phase to newcapture. Current annotations already mark partial family matching and unknown sampled timed flag.','proposed_new_admissions':0,'genesis_engine_authored':False,'postaudit_engine_authored':False,'shared_mutations':0,'GPU_build_tests':0,'bindings':bindings}
(D/'corrective_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'population':report['population'],'fdtdalreadycovered':4,'genesisheld':True,'current_samples':[(x['model'],x['current_annotations'][0]['sample_in_primary_timed_invocation'])for x in modelrows]}))
