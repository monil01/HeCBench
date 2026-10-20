#!/usr/bin/env python3
"""Recompute all twenty property gates with writes intercepted, then rehash sources/raw data."""
from pathlib import Path
import argparse,json,hashlib,importlib.util,sys
import numpy as np
P=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def module(name):
 spec=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 a=argparse.ArgumentParser();a.add_argument('--result',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();rp=z.result.resolve();r=json.load(open(rp));c=json.load(open(rp.with_name('config.json')));assert r['matched_performance_valid'];assert r['numerical_agreement_claim']is False;rawchecks=[];sourcechecks=[];seen=set();sourcehashfiles=0
 for model,spec in c['models'].items():
  for path in spec['source_dependencies']:
   p=Path(path);assert p.exists(),p
   sourcechecks.append({'model':model,'path':str(p),'sha256':sha(p)})
  for row in r['models'][model]['trials']:
   f=Path(row['raw_full_workflow_root']);recorded=json.load(open(f/'properties.json'));assert recorded['pass'];assert row['correct'];assert sha(f/'properties.json')==row['independent_property_audit_sha256'];hashes={name:sha(f/name)for name in recorded['hashes']};assert hashes==recorded['hashes'];sourcehashfiles+=len(hashes)
   m=module('check_properties'if model=='cuda'else'check_target_properties');saved_np_save=np.save;saved_write_text=Path.write_text;generated={};jsonwrites=[]
   try:
    np.save=lambda path,array:generated.setdefault(Path(path).name,np.asarray(array).copy())
    Path.write_text=lambda self,content,*args,**kwargs:jsonwrites.append((str(self),content))
    again=m.check(f)
   finally:np.save=saved_np_save;Path.write_text=saved_write_text
   assert again['pass'],(model,row['trial'],again['failures']);assert again['hashes']==recorded['hashes'];assert np.array_equal(generated['independent_NCE_loss_rows.npy'],np.load(f/'independent_NCE_loss_rows.npy'))
   artifact=row['output_artifact'];assert sha(artifact['saved_path'])==artifact['sha256'];assert artifact['count']==272624
   rawchecks.append({'model':model,'trial':row['trial'],'folder':str(f),'raw_models_sha256':{name:hashes[name]for name in ['final_syn0.f32','final_syn1.f32','final_rng.u32','callbacks.u32','vectors.bin']},'all_gates_recomputed_pass':True,'guard_bytes':again['checks']['all_5_actual_allocation_guards'if model=='cuda'else'all_6_actual_allocations_guards_5source_plus_exp_mapping']['guard_bytes'],'callbacks':again['checks']['trace_complete']['callbacks'],'normalized_loss':again['checks']['fixed_expected_loss_improvement'],'unit_gradient':again['checks']['fresh_unit_gradient_full_models'],'time_seconds':row['time_seconds'],'artifact_kind':'static_input_only_certificate_notrawmodelcomparison'})
   seen.add(hashes['final_syn0.f32']);print(model,row['trial'],'PASS',flush=True)
 root=Path(r['models']['jacc']['trials'][0]['raw_full_workflow_root']).parent
 template={}
 for name in ['jacc_template_before.f32','jacc_template_after.f32']:
  p=root/name;raw=np.fromfile(p,'<u4');assert len(raw)==418;assert np.all(raw[:17]==0x3e13579b)and np.all(raw[-17:]==0x3e13579b)and np.all(raw[17:-17]==0);template[name]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
 table=json.load(open(P/'canonical_table_provenance.json'));assert sha(table['canonical_path'])==table['sha256'];assert Path(table['canonical_path']).stat().st_size==400000000
 out={'status':'all20fullworkflow_property_gates_recomputed_pass','result':str(rp),'result_sha256':sha(rp),'config_sha256':sha(rp.with_name('config.json')),'raw_numerical_agreement_claim':False,'raw_model_identity_claim':False,'original_text8_or_fullquality_claim':False,'raw_files_fully_rehashed':sourcehashfiles,'retained_trials':20,'raw_syn0_distinct_count':len(seen),'trial_reviews':rawchecks,'current_source_dependencies':sourcechecks,'JACC_extra_template':{'actual':template,'coverage':'complete418physicalfloats atwholeprocessstart/end; notadditionalpertrialcoverage; sixCABIarrays separatelyguardcheckedall5'},'canonical_table':table,'means_last4_seconds':{m:sum(x['time_seconds']for x in v['trials'][1:])/4 for m,v in r['models'].items()},'verifier_numeric_comparisons_scope':'Exactlosslessstaticinputcertificateonly, allfullrawmodels/endogenoustrajectoryfields remainseparatepropertyobservations','no_raw_artifact_writes':True}
 z.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
